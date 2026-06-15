from __future__ import annotations

from collections import Counter

from tool_circuits.dataset import build_benchmark
from tool_circuits.io import read_jsonl
from tool_circuits.metrics import classification_metrics
from tool_circuits.patching import select_patching_rows
from tool_circuits.prompting import format_tool_prompt, normalize_label
from tool_circuits.sampling import balanced_subset
from tool_circuits.statistics import bootstrap_mean_interval, paired_permutation_pvalue


def test_normalize_label() -> None:
    assert normalize_label("web_search") == "web_search"
    assert normalize_label("`calculator`.") == "calculator"
    assert normalize_label('{"tool": "python"}') == "python"
    assert normalize_label("I cannot decide") is None


def test_prompt_contains_query() -> None:
    prompt = format_tool_prompt("Explain entropy.")
    assert "Explain entropy." in prompt
    assert "web_search, calculator, python, none" in prompt


def test_smoke_dataset_is_balanced_and_unique(tmp_path) -> None:
    counts = build_benchmark(tmp_path, "smoke", seed=7)
    assert counts["total"] == 220
    rows = list(read_jsonl(tmp_path / "tooluse_circuit_bench.jsonl"))
    assert len({row["query"] for row in rows if not row["adversarial_flag"]}) == 200
    train = list(read_jsonl(tmp_path / "train.jsonl"))
    labels = {label: sum(row["label"] == label for row in train) for label in {
        "web_search", "calculator", "python", "none"
    }}
    assert set(labels.values()) == {20}


def test_metrics() -> None:
    result = classification_metrics(
        ["web_search", "calculator", "python", "none"],
        ["web_search", "calculator", None, "python"],
    )
    assert result["accuracy"] == 0.5
    assert result["invalid_outputs"] == 1


def test_balanced_subset() -> None:
    rows = [
        {"id": f"{label}-{index}", "label": label}
        for label, count in (
            ("calculator", 6),
            ("none", 4),
            ("python", 5),
            ("web_search", 1),
        )
        for index in range(count)
    ]
    selected = balanced_subset(rows, 4)
    assert {row["label"] for row in selected} == {
        "web_search",
        "calculator",
        "python",
        "none",
    }


def test_statistics() -> None:
    mean, low, high = bootstrap_mean_interval(
        __import__("numpy").array([1.0, 2.0, 3.0]), samples=1_000
    )
    assert mean == 2.0
    assert low <= mean <= high
    assert paired_permutation_pvalue(
        __import__("numpy").array([2.0, 2.0, 2.0]),
        __import__("numpy").array([0.0, 0.0, 0.0]),
        samples=1_000,
    ) <= 0.3


def test_select_patching_rows() -> None:
    sources = [
        {"id": "c1", "label": "calculator"},
        {"id": "p1", "label": "python"},
        {"id": "c2", "label": "calculator"},
    ]
    targets = [
        {"id": "e1", "label": "calculator", "prediction": "python"},
        {"id": "ok", "label": "calculator", "prediction": "calculator"},
        {"id": "e2", "label": "calculator", "prediction": "python"},
    ]
    selected_sources, selected_targets = select_patching_rows(
        sources,
        targets,
        source_label="calculator",
        positive_label="calculator",
        negative_label="python",
        seed=42,
    )
    assert {row["id"] for row in selected_sources} == {"c1", "c2"}
    assert {row["id"] for row in selected_targets} == {"e1", "e2"}


def test_lexical_control_generator_is_balanced_and_uses_shared_frame() -> None:
    import importlib.util
    from pathlib import Path

    script = Path(__file__).parents[1] / "scripts" / "21_generate_lexical_control.py"
    spec = importlib.util.spec_from_file_location("lexical_control_generator", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rows = module.build_rows(groups=3, seed=7)
    assert len(rows) == 12
    assert Counter(row["label"] for row in rows) == Counter(
        {"web_search": 3, "calculator": 3, "python": 3, "none": 3}
    )
    assert len({row["query"] for row in rows}) == len(rows)
    assert all(row["query"].startswith(module.PREFIX) for row in rows)
    assert all(row["query"].endswith(module.SUFFIX) for row in rows)
    assert not any(
        phrase in row["query"].lower()
        for row in rows
        for phrase in ("today", "exact result", "each value")
    )
