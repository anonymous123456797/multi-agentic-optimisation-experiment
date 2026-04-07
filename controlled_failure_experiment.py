import json
import itertools
from test import evaluate_tasks, mbpp_integration

DATASET = "MBPP/mbpp_subset_50.json"  # adjust if needed

BASE_CONFIG = {
    "model_family": "qwen2.5-coder",
    "variant": "instruct",
    "temperature": 0.1,
    "top_p": 0.9,
    "num_predict": 192,
    "num_ctx": 1024,
    "exec_timeout": 10,
    "prompt_template": {
        "instruction": 0,
        "output_constraint": 0,
        "reasoning_scaffold": 0,
        "format_rule": 0,
        "test_hint": 0,
    },
}

MODEL_SIZES = ["1.5b", "7b", "32b"]
QUANTS = ["q4_K_M", "q8_0", "fp16"]
TEST_HINTS = [0, 1]

def run():
    tasks = mbpp_integration(DATASET)

    results = []

    for model_size, quant, test_hint in itertools.product(
        MODEL_SIZES, QUANTS, TEST_HINTS
    ):
        config = BASE_CONFIG.copy()
        config["model_size"] = model_size
        config["quantization"] = quant
        config["prompt_template"] = dict(BASE_CONFIG["prompt_template"])
        config["prompt_template"]["test_hint"] = test_hint

        print(f"\nRunning: {model_size}, {quant}, test_hint={test_hint}")

        summary = evaluate_tasks(config, tasks)

        results.append({
            "model_size": model_size,
            "quantization": quant,
            "test_hint": test_hint,
            "pass@1": summary["average_pass@1"],
            "runtime": summary["average_runtime"],
        })

    with open("controlled_experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nDone. Results written to controlled_experiment_results.json")


if __name__ == "__main__":
    run()
