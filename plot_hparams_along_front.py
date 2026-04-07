import os
import json
import glob
import argparse
from typing import Any, Dict, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec


# -----------------------------
# Pareto utilities
# -----------------------------
def is_pareto_efficient(points: np.ndarray) -> np.ndarray:
    """
    points: (n, m) array, all objectives assumed minimized.
    returns boolean mask of Pareto-efficient points.
    """
    n = len(points)
    efficient = np.ones(n, dtype=bool)

    for i in range(n):
        if efficient[i]:
            efficient[efficient] = (
                np.any(points[efficient] < points[i], axis=1)
                | np.all(points[efficient] == points[i], axis=1)
            )
            efficient[i] = True

    return efficient


# -----------------------------
# Loading
# -----------------------------
def load_generation_file(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {path}")
    return data


def latest_generation_file(folder: str) -> str:
    files = sorted(glob.glob(os.path.join(folder, "*_gen_*.json")))
    if not files:
        raise FileNotFoundError(f"No *_gen_*.json files found in {folder}")
    return files[-1]


# -----------------------------
# Flatten config
# -----------------------------
def flatten_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    pt = cfg.get("prompt_template", {})
    return {
        "model_size": cfg.get("model_size"),
        "quantization": cfg.get("quantization"),
        "temperature": cfg.get("temperature"),
        "top_p": cfg.get("top_p"),
        "num_predict": cfg.get("num_predict"),
        "num_ctx": cfg.get("num_ctx"),
        "instruction": pt.get("instruction"),
        "output_constraint": pt.get("output_constraint"),
        "reasoning_scaffold": pt.get("reasoning_scaffold"),
        "format_rule": pt.get("format_rule"),
        "test_hint": pt.get("test_hint", 0),
    }


# -----------------------------
# Build Pareto front
# -----------------------------
def build_front(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    points = np.array([r["objectives"][:2] for r in records], dtype=float)
    mask = is_pareto_efficient(points)

    front = [r for r, keep in zip(records, mask) if keep]

    for r in front:
        r["neg_pass_at_1"] = float(r["objectives"][0])
        r["runtime"] = float(r["objectives"][1])
        r["pass_at_1"] = -float(r["objectives"][0])
        r["flat_config"] = flatten_config(r["config"])

    # Sort from low runtime to high runtime, then by pass descending
    front.sort(key=lambda r: (r["runtime"], -r["pass_at_1"]))
    return front


# -----------------------------
# Plot helpers
# -----------------------------
def categorical_palette(values: List[Any]) -> Dict[Any, Any]:
    uniq = list(dict.fromkeys(values))
    cmap = plt.get_cmap("tab10")
    return {v: cmap(i % 10) for i, v in enumerate(uniq)}


def draw_categorical_column(ax, values: List[Any], title: str):
    palette = categorical_palette(values)
    n = len(values)

    for row, v in enumerate(values):
        y = n - 1 - row
        ax.add_patch(Rectangle((0, y), 1, 1, facecolor=palette[v], edgecolor="white"))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, n)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9, rotation=90, pad=10)

    # Legend text underneath axis
    labels = [str(v) for v in palette.keys()]
    ax.text(
        0.5, -0.03,
        "\n".join(labels),
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=6
    )


def draw_numeric_column(ax, values: List[float], title: str, color="steelblue"):
    n = len(values)
    vmin = min(values)
    vmax = max(values)

    if vmax == vmin:
        vmax = vmin + 1e-9

    for row, v in enumerate(values):
        y = n - 1 - row
        width = (v - vmin) / (vmax - vmin)
        ax.add_patch(Rectangle((0, y + 0.15), width, 0.7, facecolor=color, edgecolor="none"))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, n)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9, rotation=90, pad=10)

    ax.text(0.5, -0.03, f"{vmin:g}\n→\n{vmax:g}",
            transform=ax.transAxes, ha="center", va="top", fontsize=6)


def draw_objective_column(ax, values: List[float], title: str, color="darkorange"):
    draw_numeric_column(ax, values, title, color=color)


def draw_row_index_column(ax, n: int):
    for row in range(n):
        y = n - 1 - row
        ax.text(0.95, y + 0.5, str(row), ha="right", va="center", fontsize=7)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, n)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("cfg", fontsize=9, rotation=90, pad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)


# -----------------------------
# Main plotting function
# -----------------------------
def plot_front_columns(front: List[Dict[str, Any]], out_path: str, title: str):
    if not front:
        raise ValueError("Front is empty")

    cfgs = [r["flat_config"] for r in front]
    n = len(front)

    columns = [
        ("cfg", "index"),
        ("model_size", "cat"),
        ("quantization", "cat"),
        ("temperature", "num"),
        ("top_p", "num"),
        ("num_predict", "num"),
        ("num_ctx", "num"),
        ("instruction", "cat"),
        ("output_constraint", "cat"),
        ("reasoning_scaffold", "cat"),
        ("format_rule", "cat"),
        ("test_hint", "cat"),
        ("runtime", "obj_runtime"),
        ("pass_at_1", "obj_pass"),
    ]

    width_ratios = [
        0.5,  # cfg
        0.8,  # model_size
        1.0,  # quant
        1.0,  # temp
        1.0,  # top_p
        1.0,  # num_predict
        1.0,  # num_ctx
        0.8,  # instruction
        0.8,  # output_constraint
        0.8,  # reasoning_scaffold
        0.8,  # format_rule
        0.8,  # test_hint
        1.1,  # runtime
        1.1,  # pass
    ]

    fig = plt.figure(figsize=(18, max(4, 0.35 * n + 2)))
    gs = GridSpec(
        1,
        len(columns),
        figure=fig,
        width_ratios=width_ratios,
        wspace=0.15
    )

    for i, (key, kind) in enumerate(columns):
        ax = fig.add_subplot(gs[0, i])

        if kind == "index":
            draw_row_index_column(ax, n)
            continue

        if key in ("runtime", "pass_at_1"):
            values = [r[key] for r in front]
        else:
            values = [c[key] for c in cfgs]

        if kind == "cat":
            draw_categorical_column(ax, values, key)
        elif kind == "num":
            draw_numeric_column(ax, values, key, color="steelblue")
        elif kind == "obj_runtime":
            draw_objective_column(ax, values, key, color="darkorange")
        elif kind == "obj_pass":
            draw_objective_column(ax, values, key, color="forestgreen")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    print(f"Saved plot to {out_path}")


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None,
                        help="Specific generation JSON file")
    parser.add_argument("--folder", type=str, default="population_logs",
                        help="Folder containing generation JSONs")
    parser.add_argument("--latest", action="store_true",
                        help="Use latest generation file from folder")
    parser.add_argument("--output", type=str, default="front_columns.png",
                        help="Output image path")
    args = parser.parse_args()

    if args.file:
        path = args.file
    else:
        path = latest_generation_file(args.folder)

    records = load_generation_file(path)
    front = build_front(records)
    gen = front[0]["generation"] if front else "?"

    plot_front_columns(
        front,
        args.output,
        title=f"Hyperparameters along Pareto front (generation {gen})"
    )


if __name__ == "__main__":
    main()
