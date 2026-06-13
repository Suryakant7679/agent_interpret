from __future__ import annotations

from typing import Iterable

from . import LABELS


TOOL_INSTRUCTIONS = """You are an AI assistant with access to tools.

Available tools:
1. web_search: use for current, recent, or externally verifiable information.
2. calculator: use for exact arithmetic.
3. python: use for code execution, data analysis, or simulation.
4. none: use when no tool is needed.

User query:
{query}

Which tool should be used?
Answer with exactly one label:
web_search, calculator, python, none."""


def format_tool_prompt(query: str) -> str:
    return TOOL_INSTRUCTIONS.format(query=query.strip())


def normalize_label(text: str) -> str | None:
    cleaned = text.strip().lower()
    for char in "`\"'.,:;{}[]()":
        cleaned = cleaned.replace(char, " ")
    tokens = cleaned.split()
    exact = [label for label in LABELS if label in tokens]
    if exact:
        return exact[0]

    collapsed = cleaned.replace(" ", "_").replace("-", "_")
    for label in LABELS:
        if collapsed.startswith(label):
            return label
    return None


def validate_label_set(labels: Iterable[str]) -> None:
    invalid = sorted(set(labels) - set(LABELS))
    if invalid:
        raise ValueError(f"Unknown labels: {invalid}")
