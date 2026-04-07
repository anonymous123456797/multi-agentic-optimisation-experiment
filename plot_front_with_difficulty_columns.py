
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec


def categorical_palette(values):
    uniq = list(dict.fromkeys(values))
    cmap = plt.get_cmap("tab10")
    return {v: cmap(i % 10) for i, v in enumerate(uniq)}


def draw_categorical_column(ax, values, title):
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

    labels = [str(v) for v in palette.keys()]
    ax.text(
        0.5, -0.03,
        "\n".join(labels),
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=6
    )


def draw_numeric_column(ax, values, title, color="steelblue"):
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

    ax.text(
        0.5, -0.03,
        f"{vmin:.3g}\n→\n{vmax:.3g}",
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=6
    )


def draw_row_index_column(ax, labels):
    n = len(labels)
    for row, label in enumerate(labels):
        y = n - 1 - row
        ax.text(0.98, y + 0.5, str(label), ha="right", va="center", fontsize=7)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, n)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("cfg", fontsize=9, rotation=90, pad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)


def shorten_config_label(label: str) -> str:
    # Example: "05: 32b, q4_K_M, T=0.3, p=0.95, pred=192, ctx=1024, th=1"
    # Keep the leading index only for the row labels, because the figure gets crowded otherwise.
    return label.split(":", 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--front-csv", required=True, help="CSV like pareto_full_summary.csv")
    ap.add_argument("--difficulty-csv", required=True, help="CSV like difficulty_per_config.csv")
    ap.add_argument("--output", default="pareto_front_with_difficulty_columns.png")
    ap.add_argument("--sort", choices=["runtime", "accuracy"], default="runtime")
    args = ap.parse_args()

    front = pd.read_csv(args.front_csv)
    diff = pd.read_csv(args.difficulty_csv)

    # Pivot difficulty pass rates
    diff_pivot = diff.pivot(index="config_label", columns="difficulty", values="pass_at_1").reset_index()


    front = pd.read_csv(args.front_csv)
    diff = pd.read_csv(args.difficulty_csv)

    # older front CSVs may not have test_hint
    if "test_hint" not in front.columns:
        front["test_hint"] = 0

    # recreate the same config_label format used in analyse_difficulty_performance_full.py
    def make_label(row, idx):
        return (
            f"{idx:02d}: {row['model_size']}, {row['quantization']}, "
            f"T={row['temperature']}, p={row['top_p']}, "
            f"pred={row['num_predict']}, ctx={row['num_ctx']}, "
            f"th={row['test_hint']}"
        )

    front = front.reset_index(drop=True)
    front["config_label"] = [make_label(row, i + 1) for i, (_, row) in enumerate(front.iterrows())]

    # pivot difficulty pass rates
    diff_pivot = diff.pivot(
        index="config_label",
        columns="difficulty",
        values="pass_at_1",
    ).reset_index()

    merged = front.merge(diff_pivot, on="config_label", how="left")


    # Sort
    if args.sort == "runtime":
        merged = merged.sort_values(["average_runtime", "average_pass@1"], ascending=[True, False])
    else:
        merged = merged.sort_values(["average_pass@1", "average_runtime"], ascending=[False, True])

    # Fill missing difficulty columns if absent
    for col in ["easy", "medium", "hard"]:
        if col not in merged.columns:
            merged[col] = 0.0

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
        ("average_runtime", "obj_runtime"),
        ("average_pass@1", "obj_pass"),
        ("easy", "obj_easy"),
        ("medium", "obj_medium"),
        ("hard", "obj_hard"),
    ]

    width_ratios = [
        0.45, 0.75, 0.95, 0.95, 0.95, 0.95, 0.95,
        0.75, 0.75, 0.75, 0.75, 0.75,
        1.0, 1.0, 0.95, 0.95, 0.95
    ]

    n = len(merged)
    fig = plt.figure(figsize=(22, max(4, 0.35 * n + 2)))
    gs = GridSpec(1, len(columns), figure=fig, width_ratios=width_ratios, wspace=0.15)

    row_labels = [shorten_config_label(v) for v in merged["config_label"].tolist()]

    for i, (col, kind) in enumerate(columns):
        ax = fig.add_subplot(gs[0, i])

        if kind == "index":
            draw_row_index_column(ax, row_labels)
            continue

        values = merged[col].tolist()

        if kind == "cat":
            draw_categorical_column(ax, values, col)
        elif kind == "num":
            draw_numeric_column(ax, values, col, color="steelblue")
        elif kind == "obj_runtime":
            draw_numeric_column(ax, values, "runtime", color="darkorange")
        elif kind == "obj_pass":
            draw_numeric_column(ax, values, "overall", color="forestgreen")
        elif kind == "obj_easy":
            draw_numeric_column(ax, values, "easy", color="royalblue")
        elif kind == "obj_medium":
            draw_numeric_column(ax, values, "medium", color="mediumpurple")
        elif kind == "obj_hard":
            draw_numeric_column(ax, values, "hard", color="crimson")

    fig.suptitle("Pareto front hyperparameters with pass@1 broken down by difficulty", fontsize=14)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
