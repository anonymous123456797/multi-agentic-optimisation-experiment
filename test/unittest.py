
import pytest

import test as app



def test_build_prompt_includes_problem_text():
    task = {"prompt": "Write a function square(x) that returns x * x."}

    prompt = app.build_prompt(task)

    assert "Write a function square(x) that returns x * x." in prompt
    assert "Return only valid Python code." in prompt
    assert "Do not include markdown fences." in prompt


def test_extract_code_returns_plain_code_unchanged():
    raw = "def square(x):\n    return x * x"

    assert app.extract_code(raw) == raw


def test_extract_code_removes_think_block():
    raw = "<think>reasoning here</think>\ndef square(x):\n    return x * x"

    assert app.extract_code(raw) == "def square(x):\n    return x * x"
