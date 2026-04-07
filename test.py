import os
import re
import io
import time
import json
import logging
import traceback
import requests
import datetime
import contextlib
import multiprocessing as mp
from typing import Any, Dict, Tuple, Optional, List

from prompt_components import PROMPT_COMPONENTS


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_URL = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_TAGS_URL = f"{OLLAMA_BASE_URL}/api/tags"

FAILURES_DIR = os.getenv("FAILURES_DIR", "failures")
DEFAULT_OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "1800"))

logger = logging.getLogger(__name__)

def extract_expected_function_name(test_code: str) -> Optional[str]:
    """
    Heuristically extract the expected function name from MBPP-style test code.
    Looks for the first function call inside an assert or plain expression.
    """
    patterns = [
        r"assert\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        r"([A-Za-z_][A-Za-z0-9_]*)\s*\(",
    ]

    for pattern in patterns:
        match = re.search(pattern, test_code)
        if match:
            return match.group(1)

    return None


def build_test_hint_snippet(test_code: str, max_lines: int = 3, max_chars: int = 400) -> str:
    """
    Create a compact test snippet for prompt injection.
    """
    lines = [line.strip() for line in test_code.splitlines() if line.strip()]
    snippet = "\n".join(lines[:max_lines])

    if len(snippet) > max_chars:
        snippet = snippet[:max_chars].rstrip() + "..."

    return snippet


def build_prompt(task: Dict[str, str], config: Dict[str, Any]) -> str:
    indices = config.get("prompt_template", {})

    function_name = extract_expected_function_name(task.get("test_code", "")) or "solution"
    tests_snippet = build_test_hint_snippet(task.get("test_code", ""))

    def get_component(key: str) -> str:
        options = PROMPT_COMPONENTS[key]
        idx = indices.get(key, 0)
        if not (0 <= idx < len(options)):
            raise ValueError(
                f"prompt_template index out of range: {key}={idx} (max {len(options)-1})"
            )

        template = options[idx]
        return template.format(
            function_name=function_name,
            tests=tests_snippet,
        )

    instruction = get_component("instruction")
    output_constraint = get_component("output_constraint")
    reasoning_scaffold = get_component("reasoning_scaffold")
    format_rule = get_component("format_rule")
    test_hint = get_component("test_hint")

    parts = [
        p for p in [
            instruction,
            output_constraint,
            reasoning_scaffold,
            format_rule,
            test_hint,
        ]
        if p
    ]

    # Add a direct reminder of the required callable name
    # parts.append(f"Define a function named {function_name}.") #unneccesary?

    header = "\n".join(parts)

    return f"{header}\n\nProblem:\n{task['prompt']}".strip()


def list_local_ollama_models(timeout: int = 30) -> set[str]:
    """
    Query Ollama for locally available model names.
    """
    response = requests.get(OLLAMA_TAGS_URL, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return {m["name"] for m in data.get("models", []) if "name" in m}


def build_model_candidates(config: Dict[str, Any]) -> List[str]:
    """
    Build plausible Ollama model names from a structured config.

    Supports either:
      - explicit full model name in config["model"]
      - structured fields:
          model_family, model_size, variant, quantization

    Example:
      qwen2.5-coder + 7b + instruct + q4_K_M
      -> qwen2.5-coder:7b-instruct-q4_K_M
    """
    explicit_model = config.get("model")
    if explicit_model:
        return [explicit_model]

    family = config.get("model_family")
    size = config.get("model_size")
    variant = config.get("variant")
    quant = config.get("quantization")

    if not family or not size:
        raise ValueError(
            "Config must include either 'model' or both 'model_family' and 'model_size'."
        )

    # Build the base stem after the colon
    stem_parts = [size]
    if variant: # and variant not in ("base", "none", ""):
        stem_parts.append(variant)

    stem = "-".join(stem_parts)

    candidates = []

    # Most likely full candidate
    if quant:
        candidates.append(f"{family}:{stem}-{quant}")

    # Fallbacks in case local naming differs slightly
    candidates.append(f"{family}:{stem}")

    # Some people may tag base models without an explicit "base"
    if variant == "base":
        if quant:
            candidates.append(f"{family}:{size}-{quant}")
        candidates.append(f"{family}:{size}")

    # Also allow family only as a last-ditch fallback
    candidates.append(family)

    # Deduplicate while preserving order
    deduped = []
    seen = set()
    for c in candidates:
        if c not in seen:
            deduped.append(c)
            seen.add(c)

    return deduped


def resolve_model_name(config: Dict[str, Any]) -> str:
    """
    Resolve the concrete model name to use with Ollama.
    """
    candidates = build_model_candidates(config)

    try:
        local_models = list_local_ollama_models()
    except requests.exceptions.RequestException as exc:
        logger.warning(
            "Could not query /api/tags (%s). Falling back to first candidate: %s",
            exc,
            candidates[0],
        )
        return candidates[0]

    for candidate in candidates:
        if candidate in local_models:
            return candidate

    raise ValueError(
        "No matching local Ollama model found.\n"
        f"Tried candidates: {candidates}\n"
        f"Available local models (sample): {sorted(local_models)[:20]}"
    )


def query_ollama(
    prompt: str,
    model: str,
    config: Dict[str, Any],
) -> Tuple[str, float, Dict[str, Any]]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": config.get("temperature", 0.2),
            "top_p": config.get("top_p", 0.9),
            "num_predict": config.get("num_predict", 512),
            "num_ctx": config.get("num_ctx", 4096),
        },
    }

    timeout = int(config.get("ollama_timeout", DEFAULT_OLLAMA_TIMEOUT))

    start = time.perf_counter()
    response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    elapsed = time.perf_counter() - start

    response.raise_for_status()
    data = response.json()
    return data["response"], elapsed, data


def extract_code(raw_output: str) -> str:
    text = raw_output.strip()

    # Remove <think>...</think> if the model emits reasoning
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Handle fenced code blocks if present
    if "```" in text:
        blocks = re.findall(r"```(?:python)?\n?(.*?)```", text, flags=re.DOTALL)
        for block in blocks:
            if "def " in block or "class " in block:
                return block.strip()
        if blocks:
            return blocks[0].strip()

    return text


def _exec_code_and_tests(code_str: str, test_code: str, return_dict) -> None:
    namespace = {}
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            exec(code_str, namespace)
            exec(test_code, namespace)

        return_dict["passed"] = True
        return_dict["error"] = None
        return_dict["exception_traceback"] = None

    except Exception as e:
        return_dict["passed"] = False
        return_dict["error"] = f"{type(e).__name__}: {e}"
        return_dict["exception_traceback"] = traceback.format_exc()

    finally:
        return_dict["stdout"] = stdout_buffer.getvalue()
        return_dict["stderr"] = stderr_buffer.getvalue()


def run_tests(
    code_str: str,
    test_code: str,
    timeout_seconds: int = 10
) -> Tuple[bool, Optional[str], float, str, str, Optional[str]]:
    manager = mp.Manager()
    return_dict = manager.dict()

    process = mp.Process(
        target=_exec_code_and_tests,
        args=(code_str, test_code, return_dict),
    )

    start = time.perf_counter()
    process.start()
    process.join(timeout_seconds)
    elapsed = time.perf_counter() - start

    if process.is_alive():
        process.terminate()
        process.join()
        logger.warning("Test execution timed out after %ss", timeout_seconds)
        return (
            False,
            f"TimeoutError: code execution exceeded {timeout_seconds}s",
            elapsed,
            "",
            "",
            None,
        )

    passed = bool(return_dict.get("passed", False))
    error = return_dict.get("error", "UnknownError: no result returned")
    stdout = return_dict.get("stdout", "")
    stderr = return_dict.get("stderr", "")
    exception_traceback = return_dict.get("exception_traceback", None)

    return passed, error, elapsed, stdout, stderr, exception_traceback


def ensure_failures_dir() -> None:
    os.makedirs(FAILURES_DIR, exist_ok=True)


def sanitise_filename_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))


def dump_failed_case(result: Dict[str, Any]) -> Optional[str]:
    ensure_failures_dir()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    task_id = sanitise_filename_component(result.get("task_id", "unknown"))
    model_name = sanitise_filename_component(result.get("resolved_model", "unknown_model"))

    filename = os.path.join(
        FAILURES_DIR,
        f"failure_{timestamp}_task_{task_id}_{model_name}.json",
    )

    with open(filename, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)

    return filename


def evaluate_agent(config: Dict[str, Any], task: Dict[str, str]) -> Dict[str, Any]:
    task_id = task["task_id"]
    logger.info("Evaluating task %s", task_id)

    result = {
        "task_id": task_id,
        "config": config,
        "resolved_model": None,
        "passed": 0,
        "pass@1": 0,
        "generation_time": 0.0,
        "test_time": 0.0,
        "total_runtime": 0.0,
        "prompt_eval_count": None,
        "eval_count": None,
        "error": None,
        "exception_traceback": None,
        "raw_output": "",
        "generated_code": "",
        "stdout": "",
        "stderr": "",
        "failure_dump_file": None,
    }

    try:
        prompt = build_prompt(task, config)
        model_name = resolve_model_name(config)

        raw_output, generation_time, raw_response = query_ollama(
            prompt=prompt,
            model=model_name,
            config=config,
        )
        code = extract_code(raw_output)

        passed, error, test_time, stdout, stderr, exception_traceback = run_tests(
            code_str=code,
            test_code=task["test_code"],
            timeout_seconds=config.get("exec_timeout", 10),
        )

        result.update(
            {
                "resolved_model": model_name,
                "passed": int(passed),
                "pass@1": int(passed),
                "generation_time": round(generation_time, 4),
                "test_time": round(test_time, 4),
                "total_runtime": round(generation_time + test_time, 4),
                "prompt_eval_count": raw_response.get("prompt_eval_count"),
                "eval_count": raw_response.get("eval_count"),
                "error": error if not passed else None,
                "exception_traceback": exception_traceback if not passed else None,
                "raw_output": raw_output,
                "generated_code": code,
                "stdout": stdout,
                "stderr": stderr,
            }
        )

    except Exception as exc:
        logger.exception("Evaluation failed for task %s", task_id)
        result.update(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "exception_traceback": traceback.format_exc(),
            }
        )

    if not result["passed"]:
        logger.warning("Task %s failed", task_id)

        if result["error"]:
            logger.warning("Task %s error: %s", task_id, result["error"])

        if result["raw_output"]:
            logger.warning("Task %s raw output:\n%s", task_id, result["raw_output"])

        if result["generated_code"]:
            logger.warning("Task %s generated code:\n%s", task_id, result["generated_code"])

        if result["stdout"].strip():
            logger.warning("Task %s stdout:\n%s", task_id, result["stdout"])

        if result["stderr"].strip():
            logger.warning("Task %s stderr:\n%s", task_id, result["stderr"])

        if result["exception_traceback"]:
            logger.warning("Task %s traceback:\n%s", task_id, result["exception_traceback"])

        try:
            dump_file = dump_failed_case(result)
            result["failure_dump_file"] = dump_file
            logger.warning("Task %s failure dumped to %s", task_id, dump_file)
        except Exception:
            logger.exception("Failed to dump failure case for task %s", task_id)

    logger.info("Task %s completed: passed=%s", task_id, result["passed"])
    return result


def evaluate_tasks(config: Dict[str, Any], tasks: list[Dict[str, str]]) -> Dict[str, Any]:
    results = []

    logger.info("Evaluating %d tasks with config: %s", len(tasks), config)

    for task in tasks:
        result = evaluate_agent(config, task)
        results.append(result)

    avg_pass_at_1 = sum(r["pass@1"] for r in results) / len(results) if results else 0.0
    avg_runtime = sum(r["total_runtime"] for r in results) / len(results) if results else 0.0

    total_prompt_tokens = sum(r["prompt_eval_count"] or 0 for r in results)
    total_generated_tokens = sum(r["eval_count"] or 0 for r in results)

    failure_count = sum(1 for r in results if not r["passed"])

    logger.info(
        "Completed evaluation: tasks=%d pass@1=%.4f avg_runtime=%.4f failures=%d",
        len(results),
        avg_pass_at_1,
        avg_runtime,
        failure_count,
    )

    return {
        "config": config,
        "resolved_model": results[0]["resolved_model"] if results else None,
        "num_tasks": len(results),
        "average_pass@1": round(avg_pass_at_1, 4),
        "average_runtime": round(avg_runtime, 4),
        "total_prompt_tokens": total_prompt_tokens,
        "total_generated_tokens": total_generated_tokens,
        "failure_count": failure_count,
        "results": results,
    }


def _build_task(item):
    if not isinstance(item, dict):
        raise ValueError(f"Expected a JSON object, got {type(item).__name__}")

    prompt = item.get("text", item.get("prompt"))
    if not prompt:
        raise ValueError(
            f"Missing prompt/text field for task_id={item.get('task_id', item.get('id', 'unknown'))}"
        )

    test_parts = []

    setup_code = item.get("test_setup_code", "")
    if setup_code:
        test_parts.append(setup_code)

    for imp in item.get("test_imports", []):
        if imp:
            test_parts.append(imp)

    for test in item.get("test_list", []):
        if test:
            test_parts.append(test)

    for test in item.get("challenge_test_list", []):
        if test:
            test_parts.append(test)

    return {
        "task_id": str(item.get("task_id", item.get("id", "unknown"))),
        "prompt": prompt,
        "test_code": "\n".join(test_parts),
    }


def _load_json_array(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {filename}, got {type(data).__name__}")

    return [_build_task(item) for item in data]


def _load_jsonl(filename):
    tasks = []

    with open(filename, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {filename}") from exc

            if isinstance(item, list):
                tasks.extend(_build_task(sub_item) for sub_item in item)
            else:
                tasks.append(_build_task(item))

    return tasks


def mbpp_integration(filename):
    with open(filename, "r", encoding="utf-8") as f:
        first_non_ws = ""
        while True:
            ch = f.read(1)
            if not ch:
                break
            if not ch.isspace():
                first_non_ws = ch
                break

    if first_non_ws == "[":
        return _load_json_array(filename)

    return _load_jsonl(filename)


def print_summary(summary):
    print("\nCONFIG")
    for k, v in summary["config"].items():
        print(f"{k:18}: {v}")

    print(f"\nResolved model      : {summary.get('resolved_model')}")

    print("\nRESULTS")
    print(f"{'Task':6} {'Pass':6} {'GenTime':10} {'TestTime':10} {'Total':10}")

    for r in summary["results"]:
        print(
            f"{r['task_id']:6} "
            f"{r['passed']:6} "
            f"{r['generation_time']:<10.2f} "
            f"{r['test_time']:<10.2f} "
            f"{r['total_runtime']:<10.2f}"
        )

    print("\nAGGREGATE")
    print(f"Tasks evaluated       : {summary['num_tasks']}")
    print(f"Average pass@1        : {summary['average_pass@1']:.3f}")
    print(f"Average runtime       : {summary['average_runtime']:.2f}s")
    print(f"Total prompt tokens   : {summary['total_prompt_tokens']}")
    print(f"Total generated tokens: {summary['total_generated_tokens']}")
    print(f"Failure count         : {summary.get('failure_count', 0)}")


if __name__ == "__main__":
    config = {
        "model_family": "qwen2.5-coder",
        "model_size": "7b",
        "variant": "instruct",
        "quantization": "q4_K_M",
        "temperature": 0.1,
        "top_p": 0.9,
        "num_predict": 192,
        "num_ctx": 1024,
        "exec_timeout": 10,
        "ollama_timeout": 1800,
        "prompt_template": {
            "instruction": 0,        # "You are an expert Python programmer."
            "output_constraint": 0,  # "Return only a raw Python function..."
            "reasoning_scaffold": 0, # (none)
            "format_rule": 0,        # (none)
            "test_hint": 1,          # provide the tests
        },
    }


    toy_tasks = [
        {
            "task_id": "t1",
            "prompt": "Write a function square(x) that returns x * x.",
            "test_code": "assert square(4) == 16\nassert square(-3) == 9",
        },
        {
            "task_id": "t2",
            "prompt": "Write a function is_even(n) that returns True if n is even, otherwise False.",
            "test_code": "assert is_even(4) is True\nassert is_even(5) is False\nassert is_even(0) is True",
        },
        {
            "task_id": "t3",
            "prompt": "Write a function reverse_string(s) that returns the reverse of the input string.",
            "test_code": "assert reverse_string('abc') == 'cba'\nassert reverse_string('a') == 'a'",
        },
    ]

    experiment_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # # Toy Tests
    # summary = evaluate_tasks(config, toy_tasks)
    # print_summary(summary)
    # with open("./results/results.json", "w") as f:
    #    json.dump(summary, f, indent=2)

    # MPBB Santised Calls
    #mpbb_tasks_sanitized = mbpp_integration("MBPP/mbpp_sanitized.json")
    #mpbb_summary_sanitized = evaluate_tasks(config, mpbb_tasks_sanitized)
    #print_summary(mpbb_summary_sanitized)
    #with open("./results/results_mpbb.json", "w") as f:
    #    json.dump(mpbb_summary_sanitized, f, indent=2)


    # MPBB Calls
    mpbb_tasks = mbpp_integration("MBPP/mbpp_formatted.jsonl")
    mpbb_small = mpbb_tasks[:50]
    print("Length of Task Array: " + str(len(mpbb_small)))

    mpbb_summary = evaluate_tasks(config, mpbb_small)
    print_summary(mpbb_summary)

    os.makedirs("./results", exist_ok=True)
    filename = f"./results/results_mpbb-full-{experiment_date}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(mpbb_summary, f, indent=2, ensure_ascii=False)
