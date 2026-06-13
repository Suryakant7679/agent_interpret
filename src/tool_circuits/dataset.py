from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import LABELS
from .io import write_jsonl


@dataclass(frozen=True)
class Template:
    name: str
    category: str
    difficulty: str
    render: Callable[[random.Random], str]


CONCEPTS = [
    ("overfitting", "machine learning"),
    ("regularization", "machine learning"),
    ("Newton's second law", "physics"),
    ("photosynthesis", "biology"),
    ("binary search", "computer science"),
    ("opportunity cost", "economics"),
    ("Bayes' theorem", "statistics"),
    ("gradient descent", "optimization"),
]
SOFTWARE = ["PyTorch", "TensorFlow", "Python", "Node.js", "PostgreSQL", "CUDA"]
ORGANIZATIONS = ["OpenAI", "Microsoft", "Google", "NVIDIA", "Anthropic"]
ROLES = ["CEO", "CTO", "chief scientist"]
COUNTRIES = ["Norway", "Japan", "Canada", "Australia", "Germany"]
DATA_TOPICS = ["house prices", "exam scores", "monthly sales", "temperatures"]


def _none_explain(rng: random.Random) -> str:
    concept, _ = rng.choice(CONCEPTS)
    audience = rng.choice(
        ["a beginner", "a first-year student", "a software engineer", "a researcher"]
    )
    sentences = rng.randint(2, 30)
    analogy_id = rng.randint(1, 100)
    return (
        f"Explain {concept} in simple terms for {audience}, using {sentences} "
        f"sentences and analogy variant {analogy_id}."
    )


def _none_compare(rng: random.Random) -> str:
    pairs = [
        ("LDA", "QDA"),
        ("precision", "recall"),
        ("stack", "queue"),
        ("TCP", "UDP"),
        ("supervised learning", "unsupervised learning"),
    ]
    left, right = rng.choice(pairs)
    points = rng.randint(2, 20)
    example_id = rng.randint(1, 100)
    return (
        f"What is the conceptual difference between {left} and {right}? "
        f"Give {points} comparison points and use example variant {example_id}."
    )


def _web_current_role(rng: random.Random) -> str:
    sources = rng.randint(1, 20)
    detail = rng.randint(1, 100)
    return (
        f"Who is the current {rng.choice(ROLES)} of {rng.choice(ORGANIZATIONS)}? "
        f"Verify with {sources} recent sources and use detail level {detail}."
    )


def _web_latest_version(rng: random.Random) -> str:
    sources = rng.randint(1, 20)
    detail = rng.randint(1, 100)
    return (
        f"What is the latest stable version of {rng.choice(SOFTWARE)}? "
        f"Check {sources} official sources and use detail level {detail}."
    )


def _web_requirements(rng: random.Random) -> str:
    stay = rng.randint(3, 120)
    sources = rng.randint(1, 20)
    return (
        f"What are the current tourist visa requirements for {rng.choice(COUNTRIES)} "
        f"for an Indian citizen planning a {stay}-day stay? Cite {sources} sources."
    )


def _calculator_multiply(rng: random.Random) -> str:
    a = rng.randint(10_000, 99_999)
    b = rng.randint(100, 9_999)
    return f"What is {a} multiplied by {b}? Give the exact result."


def _calculator_interest(rng: random.Random) -> str:
    principal = rng.randrange(10_000, 100_001, 5_000)
    rate = rng.choice([4.5, 5.5, 6.75, 7.5, 8.25])
    years = rng.randint(2, 8)
    return (
        f"Calculate compound interest on INR {principal:,} at {rate}% annually "
        f"for {years} years. Give the exact amount to two decimals."
    )


def _calculator_root(rng: random.Random) -> str:
    number = rng.randint(100_000, 999_999)
    return f"What is the square root of {number}, rounded to 3 decimal places?"


def _python_statistics(rng: random.Random) -> str:
    values = [rng.randint(-100, 500) for _ in range(rng.randint(24, 40))]
    return (
        "Compute the mean, sample standard deviation, and 95% confidence interval "
        f"for this list: {values}"
    )


def _python_regression(rng: random.Random) -> str:
    xs = list(range(1, rng.randint(10, 16)))
    slope = rng.uniform(1.2, 4.8)
    ys = [round(slope * x + rng.uniform(-2, 2), 2) for x in xs]
    return (
        f"Fit an ordinary least-squares line to x={xs} and y={ys}. "
        "Report slope, intercept, and R-squared."
    )


def _python_simulation(rng: random.Random) -> str:
    flips = rng.choice([10_000, 20_000, 50_000])
    threshold = rng.randint(5, 9)
    return (
        f"Use a simulation with {flips} trials to estimate the probability of "
        f"getting at least {threshold} heads in 10 fair coin flips."
    )


STANDARD_TEMPLATES = {
    "none": [
        Template("explain_concept", "conceptual", "easy", _none_explain),
        Template("compare_concepts", "conceptual", "medium", _none_compare),
    ],
    "web_search": [
        Template("current_role", "current_facts", "easy", _web_current_role),
        Template("latest_version", "software_versions", "easy", _web_latest_version),
        Template("current_visa", "current_requirements", "medium", _web_requirements),
    ],
    "calculator": [
        Template("large_multiplication", "arithmetic", "easy", _calculator_multiply),
        Template("compound_interest", "finance_math", "medium", _calculator_interest),
        Template("square_root", "arithmetic", "medium", _calculator_root),
    ],
    "python": [
        Template("list_statistics", "data_analysis", "medium", _python_statistics),
        Template("linear_regression", "data_analysis", "hard", _python_regression),
        Template("coin_simulation", "simulation", "medium", _python_simulation),
    ],
}


def _metadata_for(label: str) -> dict[str, bool]:
    return {
        "requires_current_info": label == "web_search",
        "requires_exact_math": label == "calculator",
        "requires_execution": label == "python",
    }


def generate_standard_split(
    split: str, total: int, rng: random.Random, seen: set[str]
) -> list[dict]:
    if total % len(LABELS):
        raise ValueError(f"{split} size must be divisible by {len(LABELS)}")
    per_label = total // len(LABELS)
    rows: list[dict] = []
    for label in LABELS:
        templates = STANDARD_TEMPLATES[label]
        generated = 0
        attempts = 0
        while generated < per_label:
            attempts += 1
            if attempts > per_label * 100:
                raise RuntimeError(f"Could not generate unique samples for {label}")
            template = rng.choice(templates)
            query = template.render(rng)
            if query in seen:
                continue
            seen.add(query)
            row = {
                "id": f"{split}_{label}_{generated:05d}",
                "query": query,
                "label": label,
                "category": template.category,
                "template": template.name,
                "difficulty": template.difficulty,
                "adversarial_flag": False,
                "split": split,
                **_metadata_for(label),
            }
            rows.append(row)
            generated += 1
    rng.shuffle(rows)
    return rows


def generate_ood_split(total: int, rng: random.Random, seen: set[str]) -> list[dict]:
    if total % len(LABELS):
        raise ValueError("OOD size must be divisible by four")
    rows: list[dict] = []
    per_label = total // len(LABELS)
    renderers = {
        "web_search": lambda: (
            f"Find today's exchange rate from {rng.choice(['USD', 'EUR', 'GBP'])} "
            f"to {rng.choice(['INR', 'JPY', 'CAD'])}, rounded to "
            f"{rng.randint(2, 8)} decimals, using source set {rng.randint(1, 100)}."
        ),
        "calculator": lambda: (
            f"Evaluate ({rng.randint(200, 900)} ** 3 + {rng.randint(10, 99)}) / "
            f"{rng.randint(3, 17)} to 8 decimal places."
        ),
        "python": lambda: (
            f"Numerically integrate x^{rng.randint(2, 7)} * exp(-x) from 0 to "
            f"{rng.randint(5, 15)} with tolerance 1e-{rng.randint(4, 12)} and "
            f"estimate the integration error using method variant {rng.randint(1, 20)}."
        ),
        "none": lambda: (
            f"Why is {rng.choice(['cross-validation', 'normalization', 'recursion', 'entropy'])} "
            f"useful? Answer conceptually in {rng.randint(2, 30)} sentences using "
            f"example variant {rng.randint(1, 100)}."
        ),
    }
    for label in LABELS:
        made = 0
        while made < per_label:
            query = renderers[label]()
            if query in seen:
                continue
            seen.add(query)
            rows.append(
                {
                    "id": f"ood_{label}_{made:05d}",
                    "query": query,
                    "label": label,
                    "category": "ood_template",
                    "template": "held_out",
                    "difficulty": "hard",
                    "adversarial_flag": False,
                    "split": "ood_test",
                    **_metadata_for(label),
                }
            )
            made += 1
    rng.shuffle(rows)
    return rows


def generate_adversarial_split(total: int, rng: random.Random) -> list[dict]:
    conflicts = [
        (
            "calculator",
            lambda: (
                f"Do not use a calculator. What is {rng.randint(10_000, 99_999)} "
                f"multiplied by {rng.randint(1_000, 9_999)} exactly?"
            ),
            "calculator_denial",
        ),
        (
            "none",
            lambda: (
                f"Use web search to explain {rng.choice(CONCEPTS)[0]} conceptually "
                f"in {rng.randint(2, 30)} sentences using example variant "
                f"{rng.randint(1, 100)}."
            ),
            "unnecessary_web_request",
        ),
        (
            "web_search",
            lambda: (
                f"Use Python to tell me the current {rng.choice(ROLES)} of "
                f"{rng.choice(ORGANIZATIONS)} and verify against "
                f"{rng.randint(1, 20)} current sources."
            ),
            "wrong_python_request",
        ),
        (
            "python",
            lambda: (
                "Do not execute code. Fit a regression and report R-squared for "
                f"x={list(range(12))}, y={[rng.randint(0, 40) for _ in range(12)]}."
            ),
            "python_denial",
        ),
    ]
    rows: list[dict] = []
    seen: set[str] = set()
    index = 0
    while len(rows) < total:
        label, renderer, template = conflicts[index % len(conflicts)]
        index += 1
        query = renderer()
        if query in seen:
            continue
        seen.add(query)
        rows.append(
            {
                "id": f"adversarial_{len(rows):05d}",
                "query": query,
                "label": label,
                "category": "instruction_conflict",
                "template": template,
                "difficulty": "hard",
                "adversarial_flag": True,
                "split": "adversarial_test",
                **_metadata_for(label),
            }
        )
    rng.shuffle(rows)
    return rows


PROFILES = {
    "smoke": {
        "train": 80,
        "val": 40,
        "test": 40,
        "ood_test": 40,
        "adversarial_test": 20,
    },
    "full": {
        "train": 4_000,
        "val": 1_000,
        "test": 1_000,
        "ood_test": 1_000,
        "adversarial_test": 500,
    },
}


def build_benchmark(output_dir: str | Path, profile: str, seed: int) -> dict[str, int]:
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile {profile!r}; choose from {sorted(PROFILES)}")
    output = Path(output_dir)
    rng = random.Random(seed)
    seen: set[str] = set()
    counts = PROFILES[profile]
    all_rows: list[dict] = []
    result: dict[str, int] = {}

    for split in ("train", "val", "test"):
        rows = generate_standard_split(split, counts[split], rng, seen)
        write_jsonl(output / f"{split}.jsonl", rows)
        all_rows.extend(rows)
        result[split] = len(rows)

    ood = generate_ood_split(counts["ood_test"], rng, seen)
    adversarial = generate_adversarial_split(counts["adversarial_test"], rng)
    write_jsonl(output / "ood_test.jsonl", ood)
    write_jsonl(output / "adversarial_test.jsonl", adversarial)
    all_rows.extend(ood)
    all_rows.extend(adversarial)
    result["ood_test"] = len(ood)
    result["adversarial_test"] = len(adversarial)
    write_jsonl(output / "tooluse_circuit_bench.jsonl", all_rows)
    result["total"] = len(all_rows)
    return result
