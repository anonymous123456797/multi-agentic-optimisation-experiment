import os
import re
import glob
import json
import argparse
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd

from test import evaluate_tasks, mbpp_integration


def is_pareto_efficient(points):
    """
    points: list of objective vectors, assuming minimisation of all objectives
    returns a boolean mask of Pareto-efficient points
    """
    n = len(points)
    efficient = [True] * n

    for i in range(n):
        if not efficient[i]:
            continue

        for j in range(n):
            if i == j or not efficient[j]:
                continue

            # j dominates i if j is no worse in all objectives
            # and strictly better in at least one
            no_worse = all(points[j][k] <= points[i][k] for k in range(len(points[i])))
            strictly_better = any(points[j][k] < points[i][k] for k in range(len(points[i])))

            if no_worse and strictly_better:
                efficient[i] = False
                break

    return efficient


def find_latest_generation_json(population_dir: str) -> str:
    files = sorted(glob.glob(os.path.join(population_dir, "*_gen_*.json")))
    if not files:
        raise FileNotFoundError(f"No generation JSON files found in {population_dir}")
    return files[-1]


def load_generation_records(filename: str) -> List[Dict[str, Any]]:
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {filename}, got {type(data).__name__}")

    for i, row in enumerate(data):
        if "config" not in row or "objectives" not in row:
            raise ValueError(f"Record {i} in {filename} is missing 'config' or 'objectives'")

    return data


def extract_pareto_configs(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    points = [row["objectives"] for row in records]
    mask = is_pareto_efficient(points)

    pareto_records = [row for row, keep in zip(records, mask) if keep]

    # Deduplicate by config JSON representation
    seen = set()
    unique_configs = []

    for row in pareto_records:
        cfg = row["config"]
        key = json.dumps(cfg, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique_configs.append(cfg)

    return unique_configs


def save_pareto_configs(configs: List[Dict[str, Any]], out_file: str) -> None:
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(configs, f, indent=2, ensure_ascii=False)


def safe_model_label(config: Dict[str, Any]) -> str:
    parts = [
        config.get("model_family", "model"),
        config.get("model_size", "size"),
        config.get("variant", "variant"),
        config.get("quantization", "quant"),
        f"temp{config.get('temperature', 'x')}",
        f"top{config.get('top_p', 'x')}",
        f"pred{config.get('num_predict', 'x')}",
        f"ctx{config.get('num_ctx', 'x')}",
        f"th{config.get('prompt_template', {}).get('test_hint', 0)}",
    ]
    label = "_".join(str(p) for p in parts)
    return re.sub(r"[^A-Za-z0-9._-]+", "_", label)


def rerun_configs_on_full_mbpp(
    configs: List[Dict[str, Any]],
    dataset_file: str,
    out_dir: str,
) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)

    tasks = mbpp_integration(dataset_file)
    print(f"Loaded {len(tasks)} MBPP tasks from {dataset_file}")

    summary_rows = []

    for i, config in enumerate(configs, start=1):
        print(f"\n=== Evaluating Pareto config {i}/{len(configs)} ===")
        print(json.dumps(config, indent=2))

        summary = evaluate_tasks(config, tasks)

        label = safe_model_label(config)
        summary_path = os.path.join(out_dir, f"full_eval_{i:02d}_{label}.json")

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        summary_rows.append(
            {
                "config_id": i,
                "model_family": config.get("model_family"),
                "model_size": config.get("model_size"),
                "variant": config.get("variant"),
                "quantization": config.get("quantization"),
                "temperature": config.get("temperature"),
                "top_p": config.get("top_p"),
                "num_predict": config.get("num_predict"),
                "num_ctx": config.get("num_ctx"),
                "instruction": config.get("prompt_template", {}).get("instruction"),
                "output_constraint": config.get("prompt_template", {}).get("output_constraint"),
                "reasoning_scaffold": config.get("prompt_template", {}).get("reasoning_scaffold"),
                "format_rule": config.get("prompt_template", {}).get("format_rule"),
                "test_hint": config.get("prompt_template", {}).get("test_hint", 0),
                "average_pass@1": summary.get("average_pass@1"),
                "average_runtime": summary.get("average_runtime"),
                "failure_count": summary.get("failure_count"),
                "resolved_model": summary.get("resolved_model"),
                "json_file": summary_path,
            }
        )

    df = pd.DataFrame(summary_rows)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--population-json",
        type=str,
        default=None,
        help="Specific generation JSON file to use. If omitted, the latest file in --population-dir is used.",
    )
    parser.add_argument(
        "--population-dir",
        type=str,
        default="population_logs",
        help="Directory containing generation JSON logs.",
    )
    parser.add_argument(
        "--mbpp-file",
        type=str,
        default="MBPP/mbpp_formatted.jsonl",
        help="Full MBPP file to evaluate against.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for full Pareto reruns.",
    )

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or f"pareto_full_eval_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    population_json = args.population_json
    if population_json is None:
        population_json = find_latest_generation_json(args.population_dir)

    print(f"Using generation file: {population_json}")

    records = load_generation_records(population_json)
    configs = extract_pareto_configs(records)

    print(f"Found {len(configs)} unique Pareto-front configs")

    pareto_config_file = os.path.join(out_dir, "pareto_front_configs.json")
    save_pareto_configs(configs, pareto_config_file)

    df = rerun_configs_on_full_mbpp(
        configs=configs,
        dataset_file=args.mbpp_file,
        out_dir=out_dir,
    )

    csv_file = os.path.join(out_dir, "pareto_full_summary.csv")
    df.to_csv(csv_file, index=False)

    print(f"\nSaved Pareto configs to: {pareto_config_file}")
    print(f"Saved summary CSV to:    {csv_file}")
    print("\nSummary:")
    print(df.sort_values(["average_pass@1", "average_runtime"], ascending=[False, True]).to_string(index=False))


if __name__ == "__main__":
    main()
