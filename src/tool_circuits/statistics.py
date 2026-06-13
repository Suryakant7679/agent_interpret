from __future__ import annotations

import numpy as np


def bootstrap_mean_interval(
    values: np.ndarray,
    seed: int = 42,
    samples: int = 10_000,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("Cannot bootstrap an empty array")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, values.size), replace=True).mean(axis=1)
    alpha = (1 - confidence) / 2
    return (
        float(values.mean()),
        float(np.quantile(draws, alpha)),
        float(np.quantile(draws, 1 - alpha)),
    )


def paired_permutation_pvalue(
    treatment: np.ndarray,
    control: np.ndarray,
    seed: int = 42,
    samples: int = 100_000,
) -> float:
    differences = np.asarray(treatment, dtype=np.float64) - np.asarray(
        control, dtype=np.float64
    )
    if differences.shape == () or differences.size == 0:
        raise ValueError("Paired arrays must be non-empty")
    observed = abs(float(differences.mean()))
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(samples, differences.size))
    null = np.abs((signs * differences).mean(axis=1))
    return float((np.count_nonzero(null >= observed) + 1) / (samples + 1))
