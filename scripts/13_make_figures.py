#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Create paper-ready core figures")
    parser.add_argument("--residual-probe", type=Path, required=True)
    parser.add_argument("--mlp-probe", type=Path, required=True)
    parser.add_argument("--patching", type=Path, required=True)
    parser.add_argument("--directions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Install matplotlib: pip install matplotlib") from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    residual = json.loads(args.residual_probe.read_text(encoding="utf-8"))
    mlp = json.loads(args.mlp_probe.read_text(encoding="utf-8"))
    patching = json.loads(args.patching.read_text(encoding="utf-8"))

    fig, axis = plt.subplots(figsize=(7, 4))
    axis.plot(
        [row["layer"] for row in residual],
        [row["macro_f1"] for row in residual],
        marker="o",
        markersize=3,
        label="Residual stream",
    )
    axis.plot(
        [row["layer"] for row in mlp],
        [row["macro_f1"] for row in mlp],
        marker="s",
        markersize=3,
        label="MLP output",
    )
    axis.axhline(0.25, color="gray", linestyle="--", linewidth=1, label="Chance")
    axis.set(xlabel="Layer", ylabel="OOD macro-F1", ylim=(0, 1.05))
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "layerwise_probe.pdf")
    fig.savefig(args.output_dir / "layerwise_probe.png", dpi=300)
    plt.close(fig)

    summary = patching["summary"]
    fig, axis = plt.subplots(figsize=(7, 4))
    layers = np.asarray([row["layer"] for row in summary])
    effects = np.asarray([row["mean_effect"] for row in summary])
    low = np.asarray([row.get("effect_ci95", [row["mean_effect"]] * 2)[0] for row in summary])
    high = np.asarray([row.get("effect_ci95", [row["mean_effect"]] * 2)[1] for row in summary])
    axis.plot(layers, effects, marker="o", markersize=3)
    axis.fill_between(layers, low, high, alpha=0.2)
    axis.axhline(0, color="black", linewidth=1)
    axis.set(xlabel="Patched layer", ylabel="Calculator minus Python logit effect")
    fig.tight_layout()
    fig.savefig(args.output_dir / "activation_patching.pdf")
    fig.savefig(args.output_dir / "activation_patching.png", dpi=300)
    plt.close(fig)

    cosines = np.load(args.directions)
    labels = ["web_search", "calculator", "python", "none"]
    fig, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(cosines[-1], vmin=-1, vmax=1, cmap="coolwarm")
    axis.set_xticks(range(4), labels, rotation=30, ha="right")
    axis.set_yticks(range(4), labels)
    fig.colorbar(image, ax=axis, label="Cosine similarity")
    fig.tight_layout()
    fig.savefig(args.output_dir / "tool_direction_geometry.pdf")
    fig.savefig(args.output_dir / "tool_direction_geometry.png", dpi=300)
    plt.close(fig)
    print(f"Saved figures to {args.output_dir}")


if __name__ == "__main__":
    main()
