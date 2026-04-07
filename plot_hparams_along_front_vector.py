import os
import argparse
import matplotlib as mpl

# Vector-friendly output settings
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["figure.autolayout"] = False

import json
import glob
from typing import Any, Dict, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec


NUM_COLOR = "#4C72B0"
RUNTIME_COLOR = "#DD8452"
PASS_COLOR = "#55A868"


def format_range(vmin, vmax, title):
    if title == "num_ctx":
        return f"{int(vmin)}\n→\n{int(vmax)}"
    return f"{vmin:.3g}\n→\n{vmax:.3g}"


def is_pareto_efficient(points: np.ndarray) -> np.ndarray:
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


def build_front(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    points = np.array([r["objectives"][:2] for r in records], dtype=float)
    mask = is_pareto_efficient(points)
    front = [r for r, keep in zip(records, mask) if keep]

    for r in front:
        r["neg_pass_at_1"] = float(r["objectives"][0])
        r["runtime"] = float(r["objectives"][1])
        r["pass_at_1"] = -float(r["objectives"][0])
        r["flat_config"] = flatten_config(r["config"])

    front.sort(key=lambda r: (r["runtime"], -r["pass_at_1"]))
    return front


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
    ax.set_title(title, fontsize=12, rotation=90, pad=10)
    labels = [str(v) for v in palette.keys()]
    ax.text(0.5, -0.03, "\n".join(labels),
            transform=ax.transAxes, ha="center", va="top", fontsize=12)


def draw_numeric_column(ax, values: List[float], title: str, color=NUM_COLOR):
    n = len(values)
    vmin = min(values)
    vmax = max(values)
    if vmax == vmin:
        vmax = vmin + 1e-9

    for row, v in enumerate(values):
        y = n - 1 - row
        width = (v - vmin) / (vmax - vmin)
        ax.add_patch(Rectangle((0, y + 0.1), width, 0.8, facecolor=color, edgecolor="none"))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, n)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=12, rotation=90, pad=10)
    ax.text(0.5, -0.03, format_range(vmin, vmax, title),
            transform=ax.transAxes, ha="center", va="top", fontsize=12)


def draw_objective_column(ax, values: List[float], title: str, color=RUNTIME_COLOR):
    draw_numeric_column(ax, values, title, color=color)


def draw_row_index_column(ax, n: int):
    for row in range(n):
        y = n - 1 - row
        ax.text(0.95, y + 0.5, str(row), ha="right", va="center", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("cfg", fontsize=12, rotation=90, pad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)


def save_figure(fig, out_path: str):
    ext = os.path.splitext(out_path)[1].lower()
    save_kwargs = {"bbox_inches": "tight"}
    if ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        save_kwargs["dpi"] = 300
    fig.savefig(out_path, **save_kwargs)


def plot_front_columns(front: List[Dict[str, Any]], out_path: str):
    if not front:
        raise ValueError("Front is empty")

    cfgs = [r["flat_config"] for r in front]
    n = len(front)

    columns = [
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

    width_ratios = [0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.8, 0.8, 1.1, 1.1]

    fig = plt.figure(figsize=(18, max(4, 0.35 * n + 2)))
    gs = GridSpec(1, len(columns), figure=fig, width_ratios=width_ratios, wspace=0.08)

    for i, (key, kind) in enumerate(columns):
        ax = fig.add_subplot(gs[0, i])

        values = [r[key] for r in front] if key in ("runtime", "pass_at_1") else [c[key] for c in cfgs]

        if kind == "cat":
            draw_categorical_column(ax, values, key)
        elif kind == "num":
            draw_numeric_column(ax, values, key, color=NUM_COLOR)
        elif kind == "obj_runtime":
            draw_objective_column(ax, values, key, color=RUNTIME_COLOR)
        elif kind == "obj_pass":
            draw_objective_column(ax, values, key, color=PASS_COLOR)

    fig.tight_layout()
    save_figure(fig, out_path)
    print(f"Saved plot to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None, help="Specific generation JSON file")
    parser.add_argument("--folder", type=str, default="population_logs", help="Folder containing generation JSONs")
    parser.add_argument("--latest", action="store_true", help="Use latest generation file from folder")
    parser.add_argument("--output", type=str, default="front_columns.pdf", help="Output image path")
    args = parser.parse_args()

    path = args.file if args.file else latest_generation_file(args.folder)
    records = load_generation_file(path)
    front = build_front(records)
    plot_front_columns(front, args.output)


if __name__ == "__main__":
    main()
