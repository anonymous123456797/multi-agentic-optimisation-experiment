# prompt_components.py

PROMPT_COMPONENTS = {
    "instruction": [
        "You are an expert Python programmer.",
        "Write correct Python code.",
        "You are a neutral code generator.",
    ],
    "output_constraint": [
        "Return only valid Python code defining exactly one top-level function named {function_name}. No explanation.",
        "Return only the final function definition named {function_name}.",
        "Return only raw Python code for a function named {function_name}.",
    ],
    "reasoning_scaffold": [
        "",
        "Think step by step before writing the function.",
        "Plan your approach, then write the code.",
    ],
    "format_rule": [
        "",
        "Do not include markdown fences or backticks.",
        "Do not include any text other than the function.",
    ],
    "test_hint": [
        "",
        "The code should pass these tests:\n{tests}",
    ],
}
