import logging
import os

import json
from datetime import datetime
from pymoo.core.callback import Callback

import matplotlib.pyplot as plt
import numpy as np
from pymoo.core.callback import Callback
from pymoo.operators.sampling.lhs import LHS
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter

from test import evaluate_tasks, mbpp_integration

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

MODEL_CONFIG_SPECS = {
        "model_family": "qwen2.5-coder",
        "model_size": ["1.5b", "7b", "32b"],
        "variant": "instruct",
        "quantization": ["q4_K_M", "q8_0", "fp16"],
        "temperature": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "top_p": [0.0, 0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98],
        "num_predict": [96, 128, 192, 256, 384, 512],
        "num_ctx": [256, 384, 512, 768, 1024, 2048, 4096, 8192, 16384],
        "exec_timeout": 10,
        "prompt_template": {
            "instruction": [0,1,2],        # "You are an expert Python programmer." # [0,1,2]
            "output_constraint": [0,1,2],  # "Return only a raw Python function..." #[0,1,2]
            "reasoning_scaffold": [0,1,2], # (none) # [0,1,2]
            "format_rule": [0,1,2],       # (none) # [0,1,2]
            "test_hint": [0, 1],
        },
    }


class ResultSelectionProblem(ElementwiseProblem):
    """
    Decision variables are all integer indices into MODEL_CONFIG_SPECS.

    Objectives:
        1) maximize pass@1  -> minimize negative pass@1
        2) minimize runtime
    """

    def __init__(self, tasks: list[dict]):
        self.tasks = tasks
        self.specs = MODEL_CONFIG_SPECS

        super().__init__(
            n_var=11,
            n_obj=2,
            n_ieq_constr=0,
            xl=np.array([0] * 11, dtype=int),
            xu=np.array(
                [
                    len(self.specs["model_size"]) - 1,
                    len(self.specs["quantization"]) - 1,
                    len(self.specs["temperature"]) - 1,
                    len(self.specs["top_p"]) - 1,
                    len(self.specs["num_predict"]) - 1,
                    len(self.specs["num_ctx"]) - 1,
                    len(self.specs["prompt_template"]["instruction"]) - 1,
                    len(self.specs["prompt_template"]["output_constraint"]) - 1,
                    len(self.specs["prompt_template"]["reasoning_scaffold"]) - 1,
                    len(self.specs["prompt_template"]["format_rule"]) - 1,
                    len(self.specs["prompt_template"]["test_hint"]) - 1,
                ],
                dtype=int,
            ),
            vtype=int,
        )

    def _idx(self, x_arr: np.ndarray, i: int) -> int:
        lower = int(self.xl[i])
        upper = int(self.xu[i])
        return int(np.clip(x_arr[i], lower, upper))

    def decode_config(self, x) -> dict:
        x_arr = np.asarray(x, dtype=int).ravel()

        return {
            "model_family": self.specs["model_family"],
            "model_size": self.specs["model_size"][self._idx(x_arr, 0)],
            "variant": self.specs["variant"],
            "quantization": self.specs["quantization"][self._idx(x_arr, 1)],
            "temperature": self.specs["temperature"][self._idx(x_arr, 2)],
            "top_p": self.specs["top_p"][self._idx(x_arr, 3)],
            "num_predict": self.specs["num_predict"][self._idx(x_arr, 4)],
            "num_ctx": self.specs["num_ctx"][self._idx(x_arr, 5)],
            "exec_timeout": self.specs["exec_timeout"],
            "prompt_template": {
                "instruction": self.specs["prompt_template"]["instruction"][self._idx(x_arr, 6)],
                "output_constraint": self.specs["prompt_template"]["output_constraint"][self._idx(x_arr, 7)],
                "reasoning_scaffold": self.specs["prompt_template"]["reasoning_scaffold"][self._idx(x_arr, 8)],
                "format_rule": self.specs["prompt_template"]["format_rule"][self._idx(x_arr, 9)],
                "test_hint": self.specs["prompt_template"]["test_hint"][self._idx(x_arr, 10)],
            },
        }

    def _evaluate(self, x, out, *args, **kwargs):
        config = self.decode_config(x)
        summary = evaluate_tasks(config, self.tasks)

        out["F"] = np.array(
            [
                -summary.get("average_pass@1", 0.0),
                summary.get("average_runtime", 0.0),
            ]
        )

    def _value_to_index(self, options, value, name: str) -> int:
        try:
            return options.index(value)
        except ValueError as exc:
            raise ValueError(f"Value {value!r} not found in spec '{name}': {options}") from exc

    def encode_config(self, config: dict) -> np.ndarray:
        pt = config["prompt_template"]

        x = np.array(
            [
                self._value_to_index(self.specs["model_size"], config["model_size"], "model_size"),
                self._value_to_index(self.specs["quantization"], config["quantization"], "quantization"),
                self._value_to_index(self.specs["temperature"], config["temperature"], "temperature"),
                self._value_to_index(self.specs["top_p"], config["top_p"], "top_p"),
                self._value_to_index(self.specs["num_predict"], config["num_predict"], "num_predict"),
                self._value_to_index(self.specs["num_ctx"], config["num_ctx"], "num_ctx"),
                self._value_to_index(self.specs["prompt_template"]["instruction"], pt["instruction"], "prompt_template.instruction"),
                self._value_to_index(self.specs["prompt_template"]["output_constraint"], pt["output_constraint"], "prompt_template.output_constraint"),
                self._value_to_index(self.specs["prompt_template"]["reasoning_scaffold"], pt["reasoning_scaffold"], "prompt_template.reasoning_scaffold"),
                self._value_to_index(self.specs["prompt_template"]["format_rule"], pt["format_rule"], "prompt_template.format_rule"),
                self._value_to_index(self.specs["prompt_template"]["test_hint"], pt.get("test_hint", 0), "prompt_template.test_hint"),
            ],
            dtype=int,
        )

        return np.clip(x, self.xl, self.xu)


def load_seed_configs(filename: str) -> list[dict]:
    with open(filename, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {filename}, got {type(data).__name__}")

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Seed #{i} in {filename} is not a JSON object")

    return data


def unique_rows(a: np.ndarray) -> np.ndarray:
    seen = set()
    rows = []
    for row in a:
        key = tuple(int(v) for v in row)
        if key not in seen:
            seen.add(key)
            rows.append(row)
    return np.array(rows, dtype=int) if rows else np.empty((0, a.shape[1]), dtype=int)


def build_seeded_lhs_population(
    problem,
    pop_size: int,
    seed_file: str,
    seed: int = 1,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    seed_configs = load_seed_configs(seed_file)
    for i, cfg in enumerate(seed_configs):
        try:
            problem.encode_config(cfg)
        except Exception as exc:
            raise ValueError(f"Invalid seed config at index {i}: {cfg}") from exc

    seed_vectors = np.array([problem.encode_config(cfg) for cfg in seed_configs], dtype=int)
    seed_vectors = unique_rows(seed_vectors)

    if len(seed_vectors) >= pop_size:
        return seed_vectors[:pop_size]

    n_remaining = pop_size - len(seed_vectors)
    lhs_n = max(n_remaining * 3, n_remaining + 10)

    lhs_sampler = LHS()
    lhs_X = lhs_sampler.do(problem, lhs_n).get("X")
    lhs_X = np.rint(lhs_X).astype(int)
    lhs_X = np.clip(lhs_X, problem.xl, problem.xu)
    lhs_X = unique_rows(lhs_X)

    seed_keys = {tuple(row.tolist()) for row in seed_vectors}
    lhs_rows = [row for row in lhs_X if tuple(row.tolist()) not in seed_keys]
    lhs_X = np.array(lhs_rows, dtype=int) if lhs_rows else np.empty((0, problem.n_var), dtype=int)

    X = np.vstack([seed_vectors, lhs_X]) if len(lhs_X) > 0 else seed_vectors

    while len(X) < pop_size:
        extra = rng.integers(
            low=problem.xl,
            high=problem.xu + 1,
            size=(pop_size - len(X), problem.n_var),
        )
        X = unique_rows(np.vstack([X, extra]))

    return X[:pop_size]


class PopulationLogger(Callback):
    def __init__(self, problem, out_dir="population_logs"):
        super().__init__()
        self.problem = problem
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def notify(self, algorithm):
        gen = algorithm.n_gen
        pop = algorithm.pop

        X = pop.get("X")
        F = pop.get("F")

        records = []
        for i, (x, f) in enumerate(zip(X, F)):
            x_list = np.asarray(x, dtype=int).tolist()

            record = {
                "generation": int(gen),
                "individual": int(i),
                "decision_vector": x_list,
                "objectives": np.asarray(f, dtype=float).tolist(),
                "config": self.problem.decode_config(x),
            }
            records.append(record)

        filename = os.path.join(
            self.out_dir,
            f"{self.run_id}_gen_{gen:03d}.json",
        )

        with open(filename, "w", encoding="utf-8") as fh:
            json.dump(records, fh, indent=2)

        print(f"[PopulationLogger] Saved generation {gen} to {filename}")


def main():
    mpbb_tasks = mbpp_integration("strat_sample/gemini2/mbpp_subset_50.json") #"MBPP/mbpp_formatted.jsonl")
    mpbb_small = mpbb_tasks #[:1] # First tasks
    TASKS = mpbb_small
    print("Length of Task Array: " + str(len(mpbb_small)))

    # Build the optimization problem
    problem = ResultSelectionProblem(mpbb_small)

    # Configure the optimizer
    ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=12)

    pop_size = 20
    seed_file = "seed_configs.json"

    initial_sampling = build_seeded_lhs_population(
        problem=problem,
        pop_size=pop_size,
        seed_file=seed_file,
        seed=1,
    )

    print("\nInitial population:")
    for i, x in enumerate(initial_sampling):
        print(f"{i:2d}: {x.tolist()} -> {problem.decode_config(x)}")

    algorithm = NSGA3(
        ref_dirs=ref_dirs,
        pop_size=pop_size,
        sampling=initial_sampling,
        crossover=SBX(prob=0.9, eta=15, vtype=float, repair=RoundingRepair()),
        mutation=PM(eta=20, vtype=float, repair=RoundingRepair()),
    )

    termination = get_termination("n_gen", 75)

    callback = PopulationLogger(problem, out_dir="population_logs")

    # Run optimization
    result = minimize(
        problem,
        algorithm,
        termination,
        seed=1,
        verbose=False,
        callback=callback,
    )

    print("\nObjective values:")
    print(result.F)
    print("\nDecision vectors:")
    print(result.X)

    # Show the Pareto front
    plot = Scatter()
    plot.add(result.F)
    plot.title = "Pareto Front"
    plot.xlabel = "Objective 1: -pass@1"
    plot.ylabel = "Objective 2: runtime"
    plot.show()

    plt.figure()
    plt.scatter(result.F[:, 0], result.F[:, 1])
    plt.xlabel("Minimize: -pass@1")
    plt.ylabel("Minimize: runtime")
    plt.title("Pareto Front")
    plt.show()


if __name__ == "__main__":
    main()

