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
        f"{vmin:g}\n→\n{vmax:g}",
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=6
    )


def draw_row_index_column(ax, n):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to pareto_full_summary.csv")
    parser.add_argument("--output", default="pareto_full_columns.png")
    parser.add_argument("--sort", choices=["runtime", "accuracy"], default="runtime")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    if args.sort == "runtime":
        df = df.sort_values(["average_runtime", "average_pass@1"], ascending=[True, False])
    else:
        df = df.sort_values(["average_pass@1", "average_runtime"], ascending=[False, True])

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
    ]

    width_ratios = [0.5, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.8, 0.8, 1.1, 1.1]

    n = len(df)
    fig = plt.figure(figsize=(18, max(4, 0.35 * n + 2)))
    gs = GridSpec(1, len(columns), figure=fig, width_ratios=width_ratios, wspace=0.15)

    for i, (col, kind) in enumerate(columns):
        ax = fig.add_subplot(gs[0, i])

        if kind == "index":
            draw_row_index_column(ax, n)
            continue

        values = df[col].tolist()

        if kind == "cat":
            draw_categorical_column(ax, values, col)
        elif kind == "num":
            draw_numeric_column(ax, values, col, color="steelblue")
        elif kind == "obj_runtime":
            draw_numeric_column(ax, values, col, color="darkorange")
        elif kind == "obj_pass":
            draw_numeric_column(ax, values, col, color="forestgreen")

    fig.suptitle("Hyperparameters along final Pareto front (full MBPP)", fontsize=14)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
