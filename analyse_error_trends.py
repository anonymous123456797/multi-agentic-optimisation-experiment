
import argparse
import csv
import io
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt


FAIL_RE = re.compile(
    r"failure_(\d{8}_\d{6})_(\d+)_task_([^_]+)_(.+)\.json$"
)
POP_RE = re.compile(
    r"(\d{8}_\d{6})_gen_(\d+)\.json$"
)


def parse_fail_ts(name: str):
    m = FAIL_RE.search(name)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S"), int(m.group(2))


def parse_pop_meta(name: str):
    m = POP_RE.search(name)
    if not m:
        return None
    return m.group(1), int(m.group(2))


def config_key(cfg: dict):
    pt = cfg.get("prompt_template", {})
    return (
        cfg.get("model_family"),
        cfg.get("model_size"),
        cfg.get("variant"),
        cfg.get("quantization"),
        cfg.get("temperature"),
        cfg.get("top_p"),
        cfg.get("num_predict"),
        cfg.get("num_ctx"),
        cfg.get("exec_timeout"),
        pt.get("instruction"),
        pt.get("output_constraint"),
        pt.get("reasoning_scaffold"),
        pt.get("format_rule"),
        pt.get("test_hint", 0),
    )


def classify_failure(rec: dict):
    err = (rec.get("error") or "").strip()
    raw = rec.get("raw_output") or ""
    gen = rec.get("generated_code") or ""
    tb = rec.get("exception_traceback") or ""

    if "TimeoutError" in err:
        return "timeout"
    if "SyntaxError" in err or "IndentationError" in err:
        return "syntax_error"
    if "ImportError" in err or "ModuleNotFoundError" in err:
        return "import_error"
    if "AssertionError" in err:
        return "assertion_failure"

    if "NameError" in err and "is not defined" in err:
        m = re.search(r"name '([A-Za-z_][A-Za-z0-9_]*)' is not defined", err)
        expected = m.group(1) if m else None
        funcs = re.findall(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", gen or raw)
        if expected and funcs:
            if expected in funcs:
                return "nameerror_other"
            lower_map = {f.lower(): f for f in funcs}
            if expected.lower() in lower_map:
                return "function_name_case_mismatch"
            return "function_name_mismatch"
        return "nameerror_other"

    if "TypeError" in err:
        return "signature_or_type_error"
    if "ValueError" in err or "KeyError" in err or "IndexError" in err:
        return "other_runtime_error"
    if err:
        return "other_runtime_error"
    if tb:
        return "other_runtime_error"
    return "unknown"


def load_population_run(pop_zip_path: str, run_id: Optional[str] = None):
    with zipfile.ZipFile(pop_zip_path) as zf:
        files = [n for n in zf.namelist() if n.endswith(".json") and "_gen_" in Path(n).name]

        runs = defaultdict(list)
        for name in files:
            meta = parse_pop_meta(Path(name).name)
            if meta:
                rid, gen = meta
                runs[rid].append((gen, name))

        if not runs:
            raise ValueError("No population log generation files found.")

        if run_id is None:
            run_id = max(runs.items(), key=lambda kv: len(kv[1]))[0]

        if run_id not in runs:
            raise ValueError(f"Run id {run_id} not found. Available: {sorted(runs)}")

        entries = sorted(runs[run_id], key=lambda x: x[0])

        gen_to_configs = defaultdict(list)
        cfg_first_gen = {}
        run_start = None
        run_end = None

        for gen, name in entries:
            ts = datetime.strptime(run_id, "%Y%m%d_%H%M%S")
            if run_start is None:
                run_start = ts
            run_end = ts

            with zf.open(name) as fh:
                data = json.load(fh)

            for rec in data:
                cfg = rec.get("config", {})
                ck = config_key(cfg)
                gen_to_configs[gen].append(ck)
                if ck not in cfg_first_gen:
                    cfg_first_gen[ck] = gen

        # generous window: from run timestamp to +7 days unless later refined
        window_end = run_end + timedelta(days=7)

        # try to tighten window using next run timestamp if present
        sorted_run_ids = sorted(runs.keys())
        idx = sorted_run_ids.index(run_id)
        if idx + 1 < len(sorted_run_ids):
            next_ts = datetime.strptime(sorted_run_ids[idx + 1], "%Y%m%d_%H%M%S")
            window_end = min(window_end, next_ts)

        final_gen = max(gen_to_configs) if gen_to_configs else None
        final_front_cfgs = set(gen_to_configs.get(final_gen, []))

        return {
            "run_id": run_id,
            "run_start": run_start,
            "run_end": window_end,
            "cfg_first_gen": cfg_first_gen,
            "final_front_cfgs": final_front_cfgs,
            "gens": sorted(gen_to_configs.keys()),
        }


def load_filtered_failures(fail_zip_path: str, pop_info: dict):
    rows = []
    run_start = pop_info["run_start"]
    run_end = pop_info["run_end"]
    cfg_first_gen = pop_info["cfg_first_gen"]
    final_front_cfgs = pop_info["final_front_cfgs"]

    with zipfile.ZipFile(fail_zip_path) as zf:
        files = [n for n in zf.namelist() if Path(n).name.startswith("failure_") and n.endswith(".json")]

        for name in files:
            parsed = parse_fail_ts(Path(name).name)
            if not parsed:
                continue
            ts, micros = parsed
            if not (run_start <= ts < run_end):
                continue

            with zf.open(name) as fh:
                rec = json.load(fh)

            ck = config_key(rec.get("config", {}))
            if ck not in cfg_first_gen:
                continue

            rows.append({
                "file": Path(name).name,
                "timestamp": ts.isoformat(sep=" "),
                "task_id": rec.get("task_id"),
                "generation": cfg_first_gen.get(ck),
                "is_final_pareto_config": ck in final_front_cfgs,
                "error": rec.get("error"),
                "category": classify_failure(rec),
                "model_size": rec.get("config", {}).get("model_size"),
                "quantization": rec.get("config", {}).get("quantization"),
                "temperature": rec.get("config", {}).get("temperature"),
                "top_p": rec.get("config", {}).get("top_p"),
                "num_predict": rec.get("config", {}).get("num_predict"),
                "num_ctx": rec.get("config", {}).get("num_ctx"),
                "instruction": rec.get("config", {}).get("prompt_template", {}).get("instruction"),
                "output_constraint": rec.get("config", {}).get("prompt_template", {}).get("output_constraint"),
                "reasoning_scaffold": rec.get("config", {}).get("prompt_template", {}).get("reasoning_scaffold"),
                "format_rule": rec.get("config", {}).get("prompt_template", {}).get("format_rule"),
                "test_hint": rec.get("config", {}).get("prompt_template", {}).get("test_hint", 0),
            })

    return rows


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def summary_counts(rows, by_field):
    counts = Counter(r[by_field] for r in rows)
    return [{by_field: k, "count": v} for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))]


def summary_by_generation(rows):
    gens = defaultdict(Counter)
    for r in rows:
        gens[r["generation"]][r["category"]] += 1
    out = []
    for gen in sorted(gens):
        total = sum(gens[gen].values())
        rec = {"generation": gen, "total": total}
        rec.update(gens[gen])
        out.append(rec)
    return out


def plot_generation_trends(rows, out_path: Path):
    gens = sorted({r["generation"] for r in rows if r["generation"] is not None})
    cats = sorted({r["category"] for r in rows})
    if not gens:
        return

    series = {cat: [] for cat in cats}
    for g in gens:
        bucket = Counter(r["category"] for r in rows if r["generation"] == g)
        for cat in cats:
            series[cat].append(bucket.get(cat, 0))

    plt.figure(figsize=(10, 5))
    for cat in cats:
        plt.plot(gens, series[cat], marker="o", label=cat)
    plt.xlabel("Generation")
    plt.ylabel("Failure count")
    plt.title("Failure categories by generation")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_pareto_kinds(rows, out_path: Path):
    sub = [r for r in rows if r["is_final_pareto_config"]]
    counts = Counter(r["category"] for r in sub)
    if not counts:
        return
    labels = list(counts.keys())
    vals = [counts[k] for k in labels]
    plt.figure(figsize=(8, 4))
    plt.bar(labels, vals)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Count")
    plt.title("Failure categories for final Pareto configurations")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_grouped_counts(rows, field, out_path: Path):
    groups = defaultdict(Counter)
    for r in rows:
        groups[str(r[field])][r["category"]] += 1

    group_names = sorted(groups.keys())
    cats = sorted({c for ctr in groups.values() for c in ctr.keys()})
    if not group_names or not cats:
        return

    x = list(range(len(group_names)))
    width = 0.8 / max(1, len(cats))

    plt.figure(figsize=(max(8, len(group_names) * 1.2), 5))
    for i, cat in enumerate(cats):
        vals = [groups[g].get(cat, 0) for g in group_names]
        offs = [xi - 0.4 + i * width + width / 2 for xi in x]
        plt.bar(offs, vals, width=width, label=cat)

    plt.xticks(x, group_names, rotation=30, ha="right")
    plt.ylabel("Failure count")
    plt.title(f"Failure categories by {field}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--failures", required=True, help="Path to failures.zip")
    ap.add_argument("--population-logs", required=True, help="Path to population logs zip")
    ap.add_argument("--run-id", default=None, help="Specific run id like 20260329_144102")
    ap.add_argument("--out-dir", default="error_trends", help="Output directory")
    ap.add_argument("--out-prefix", default="ga_run_only", help="Prefix for filenames")
    args = ap.parse_args()

    pop_info = load_population_run(args.population_logs, args.run_id)
    rows = load_filtered_failures(args.failures, pop_info)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = args.out_prefix
    parent = out_dir

    details_csv = parent / f"{stem}_details.csv"
    overall_csv = parent / f"{stem}_overall.csv"
    by_gen_csv = parent / f"{stem}_by_generation.csv"
    pareto_csv = parent / f"{stem}_pareto.csv"
    by_model_csv = parent / f"{stem}_by_model_size.csv"
    by_quant_csv = parent / f"{stem}_by_quantization.csv"
    by_hint_csv = parent / f"{stem}_by_test_hint.csv"

    if rows:
        write_csv(details_csv, rows, list(rows[0].keys()))
    else:
        write_csv(details_csv, [], ["file","timestamp","task_id","generation","is_final_pareto_config","error","category"])

    overall = summary_counts(rows, "category")
    write_csv(overall_csv, overall, ["category", "count"])

    by_gen = summary_by_generation(rows)
    gen_fields = sorted({k for r in by_gen for k in r.keys()}, key=lambda x: (x not in ("generation","total"), x))
    write_csv(by_gen_csv, by_gen, gen_fields)

    pareto_rows = [r for r in rows if r["is_final_pareto_config"]]
    pareto = summary_counts(pareto_rows, "category")
    write_csv(pareto_csv, pareto, ["category", "count"])

    by_model = []
    for k, ctr in sorted(defaultdict(Counter, {}).items()):
        pass

    # simple grouped csv summaries
    def grouped_summary(field):
        groups = defaultdict(Counter)
        for r in rows:
            groups[str(r[field])][r["category"]] += 1
        out = []
        for g in sorted(groups):
            rec = {field: g}
            rec.update(groups[g])
            rec["total"] = sum(groups[g].values())
            out.append(rec)
        return out

    for field, path in [("model_size", by_model_csv), ("quantization", by_quant_csv), ("test_hint", by_hint_csv)]:
        summ = grouped_summary(field)
        fields = sorted({k for r in summ for k in r.keys()}, key=lambda x: (x not in (field,"total"), x))
        write_csv(path, summ, fields)

    # plots
    plot_generation_trends(rows, parent / f"{stem}_generation_error_trends.png")
    plot_pareto_kinds(rows, parent / f"{stem}_pareto_error_kinds.png")
    plot_grouped_counts(rows, "model_size", parent / f"{stem}_failure_by_model_size.png")
    plot_grouped_counts(rows, "quantization", parent / f"{stem}_failure_by_quantization.png")
    plot_grouped_counts(rows, "test_hint", parent / f"{stem}_failure_by_test_hint.png")

    print(f"Selected GA run: {pop_info['run_id']}")
    print(f"Run window: {pop_info['run_start']} to {pop_info['run_end']}")
    print(f"Filtered failures: {len(rows)}")
    print(f"Wrote: {details_csv}")
    print(f"Wrote: {overall_csv}")
    print(f"Wrote: {by_gen_csv}")
    print(f"Wrote: {pareto_csv}")


if __name__ == "__main__":
    main()
