
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_difficulty_map(mbpp_jsonl_path: str) -> dict:
    difficulty_map = {}
    with open(mbpp_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            task_id = str(item.get("task_id", item.get("id")))
            difficulty = item.get("difficulty", "unknown")
            difficulty_map[task_id] = difficulty
    return difficulty_map


def config_label_from_summary(summary: dict, idx: int) -> str:
    cfg = summary["config"]
    pt = cfg.get("prompt_template", {})
    return (
        f"{idx:02d}: {cfg.get('model_size')}, {cfg.get('quantization')}, "
        f"T={cfg.get('temperature')}, p={cfg.get('top_p')}, "
        f"pred={cfg.get('num_predict')}, ctx={cfg.get('num_ctx')}, "
        f"th={pt.get('test_hint', 0)}"
    )


def load_eval_summaries_from_dir(dir_path: str) -> list:
    summaries = []
    dir_path = Path(dir_path)

    if not dir_path.exists():
        raise ValueError(f"Directory not found: {dir_path}")

    for path in sorted(dir_path.glob("**/*.json")):
        if path.name.startswith("full_eval_"):
            with open(path, "r", encoding="utf-8") as fh:
                summaries.append(json.load(fh))

    if not summaries:
        raise ValueError("No full_eval_*.json files found in directory.")

    return summaries


def build_long_dataframe(eval_summaries: list, difficulty_map: dict) -> pd.DataFrame:
    rows = []
    for idx, summary in enumerate(eval_summaries, start=1):
        label = config_label_from_summary(summary, idx)
        cfg = summary["config"]
        for r in summary["results"]:
            task_id = str(r["task_id"])
            difficulty = difficulty_map.get(task_id, "unknown")
            rows.append({
                "config_label": label,
                "task_id": task_id,
                "difficulty": difficulty,
                "passed": int(r.get("pass@1", r.get("passed", 0))),
                "runtime": float(r.get("total_runtime", 0.0)),
                "model_size": cfg.get("model_size"),
                "quantization": cfg.get("quantization"),
                "temperature": cfg.get("temperature"),
                "top_p": cfg.get("top_p"),
                "num_predict": cfg.get("num_predict"),
                "num_ctx": cfg.get("num_ctx"),
                "test_hint": cfg.get("prompt_template", {}).get("test_hint", 0),
            })
    return pd.DataFrame(rows)


def save_csvs(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    per_cfg = (
        df.groupby(["config_label", "difficulty"], as_index=False)["passed"]
        .mean()
        .rename(columns={"passed": "pass_at_1"})
    )
    per_cfg.to_csv(out_dir / "difficulty_per_config.csv", index=False)

    overall = (
        df.groupby("difficulty", as_index=False)["passed"]
        .mean()
        .rename(columns={"passed": "pass_at_1"})
    )
    overall.to_csv(out_dir / "difficulty_overall.csv", index=False)

    by_family = (
        df.groupby(["model_size", "quantization", "difficulty"], as_index=False)["passed"]
        .mean()
        .rename(columns={"passed": "pass_at_1"})
    )
    by_family.to_csv(out_dir / "difficulty_by_model_quant.csv", index=False)

    counts = df.groupby("difficulty", as_index=False).size().rename(columns={"size": "num_tasks"})
    counts.to_csv(out_dir / "difficulty_counts.csv", index=False)

    return per_cfg, overall, by_family, counts


def plot_overall(overall: pd.DataFrame, out_dir: Path):
    order = [d for d in ["easy", "medium", "hard", "unknown"] if d in overall["difficulty"].tolist()]
    overall = overall.set_index("difficulty").reindex(order).reset_index()

    plt.figure(figsize=(6, 4))
    plt.bar(overall["difficulty"], overall["pass_at_1"])
    plt.ylabel("pass@1")
    plt.title("Overall pass@1 by difficulty")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_dir / "difficulty_overall.png", dpi=180)
    plt.close()


def plot_per_config_grouped(per_cfg: pd.DataFrame, out_dir: Path):
    pivot = per_cfg.pivot(index="config_label", columns="difficulty", values="pass_at_1").fillna(0.0)
    cols = [c for c in ["easy", "medium", "hard", "unknown"] if c in pivot.columns]
    pivot = pivot[cols]

    n_cfg = len(pivot)
    x = range(n_cfg)
    width = 0.8 / max(1, len(cols))

    plt.figure(figsize=(max(10, n_cfg * 0.6), 5))
    for i, col in enumerate(cols):
        xs = [xi - 0.4 + i * width + width / 2 for xi in x]
        plt.bar(xs, pivot[col].tolist(), width=width, label=col)

    plt.xticks(list(x), pivot.index.tolist(), rotation=75, ha="right", fontsize=7)
    plt.ylabel("pass@1")
    plt.title("Pass@1 by difficulty for each Pareto config")
    plt.ylim(0, 1)
    plt.legend(title="Difficulty")
    plt.tight_layout()
    plt.savefig(out_dir / "difficulty_per_config_grouped.png", dpi=180)
    plt.close()


def plot_heatmap(per_cfg: pd.DataFrame, out_dir: Path):
    pivot = per_cfg.pivot(index="config_label", columns="difficulty", values="pass_at_1").fillna(0.0)
    cols = [c for c in ["easy", "medium", "hard", "unknown"] if c in pivot.columns]
    pivot = pivot[cols]

    fig, ax = plt.subplots(figsize=(6, max(4, len(pivot) * 0.35)))
    im = ax.imshow(pivot.values, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_title("Pass@1 heatmap by config and difficulty")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("pass@1")

    plt.tight_layout()
    plt.savefig(out_dir / "difficulty_heatmap.png", dpi=180)
    plt.close()


def plot_model_quant(by_family: pd.DataFrame, out_dir: Path):
    rows = []
    for (ms, q), sub in by_family.groupby(["model_size", "quantization"]):
        rec = {"group": f"{ms}-{q}"}
        for _, r in sub.iterrows():
            rec[r["difficulty"]] = r["pass_at_1"]
        rows.append(rec)

    df = pd.DataFrame(rows).fillna(0.0)
    if df.empty:
        return

    df = df.sort_values("group")
    cols = [c for c in ["easy", "medium", "hard", "unknown"] if c in df.columns]

    x = range(len(df))
    width = 0.8 / max(1, len(cols))

    plt.figure(figsize=(max(8, len(df) * 0.8), 5))
    for i, col in enumerate(cols):
        xs = [xi - 0.4 + i * width + width / 2 for xi in x]
        plt.bar(xs, df[col].tolist(), width=width, label=col)

    plt.xticks(list(x), df["group"].tolist(), rotation=45, ha="right")
    plt.ylabel("pass@1")
    plt.title("Pass@1 by difficulty grouped by model size and quantization")
    plt.ylim(0, 1)
    plt.legend(title="Difficulty")
    plt.tight_layout()
    plt.savefig(out_dir / "difficulty_by_model_quant.png", dpi=180)
    plt.close()


def write_short_report(overall: pd.DataFrame, per_cfg: pd.DataFrame, out_dir: Path):
    order = [d for d in ["easy", "medium", "hard"] if d in overall["difficulty"].tolist()]
    o = overall.set_index("difficulty")
    lines = []
    lines.append("Overall pass@1 by difficulty:")
    for d in order:
        lines.append(f"  {d}: {o.loc[d, 'pass_at_1']:.3f}")
    if all(d in o.index for d in ["easy", "medium", "hard"]):
        lines.append("")
        lines.append(
            f"Drop easy→hard: {o.loc['easy', 'pass_at_1'] - o.loc['hard', 'pass_at_1']:.3f}"
        )

    lines.append("")
    lines.append("Best config per difficulty:")
    best = per_cfg.sort_values(["difficulty", "pass_at_1"], ascending=[True, False]).groupby("difficulty").head(1)
    for _, r in best.iterrows():
        lines.append(f"  {r['difficulty']}: {r['config_label']} ({r['pass_at_1']:.3f})")

    (out_dir / "difficulty_report.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True, help="Directory of full Pareto evaluations")
    ap.add_argument("--mbpp", required=True, help="MBPP jsonl with difficulty labels")
    ap.add_argument("--out-dir", default="difficulty_analysis", help="Output directory")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    diff_map = load_difficulty_map(args.mbpp)
    summaries = load_eval_summaries_from_dir(args.eval_dir)
    df = build_long_dataframe(summaries, diff_map)

    per_cfg, overall, by_family, counts = save_csvs(df, out_dir)
    plot_overall(overall, out_dir)
    plot_per_config_grouped(per_cfg, out_dir)
    plot_heatmap(per_cfg, out_dir)
    plot_model_quant(by_family, out_dir)
    write_short_report(overall, per_cfg, out_dir)

    print(f"Wrote outputs to {out_dir}")
    print("Files:")
    for p in sorted(out_dir.iterdir()):
        print(" ", p.name)


if __name__ == "__main__":
    main()
