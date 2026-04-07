# stratified_sample.py

import json
import random
import argparse
from collections import defaultdict


def load_data(filename):
    with open(filename, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)

        # JSON array
        if first == "[":
            return json.load(f)

        # JSONL
        return [json.loads(line) for line in f if line.strip()]


def save_data(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def stratified_sample(data, fraction, seed=42, key="difficulty"):
    random.seed(seed)

    strata = defaultdict(list)

    # Group by difficulty
    for item in data:
        if key not in item:
            raise ValueError(f"Missing '{key}' in item: {item}")
        strata[item[key]].append(item)

    sampled = []

    for difficulty, items in strata.items():
        n_total = len(items)
        n_sample = max(1, int(round(n_total * fraction)))

        if n_sample > n_total:
            n_sample = n_total

        sampled_items = random.sample(items, n_sample)
        sampled.extend(sampled_items)

        print(f"{difficulty:10}: {n_sample}/{n_total}")

    random.shuffle(sampled)  # mix difficulties
    return sampled


def main():
    parser = argparse.ArgumentParser(description="Stratified sampling by difficulty")
    parser.add_argument("input", help="Input JSON/JSONL file")
    parser.add_argument("output", help="Output JSON file")
    parser.add_argument("--fraction", type=float, default=0.2, help="Fraction to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    data = load_data(args.input)
    print(f"Loaded {len(data)} items")

    sampled = stratified_sample(data, args.fraction, args.seed)

    print(f"\nSampled {len(sampled)} items total")
    save_data(sampled, args.output)


if __name__ == "__main__":
    main()
