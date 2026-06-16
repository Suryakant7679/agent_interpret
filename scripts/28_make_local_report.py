#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


LABELS = ["web_search", "calculator", "python", "none"]
COLORS = {
    "Qwen2.5-7B": "#3366cc",
    "Qwen2.5-Coder-7B": "#dc3912",
    "Mistral-7B": "#109618",
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def holm_adjust(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=np.float64)
    running = 0.0
    count = len(pvalues)
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * pvalues[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def save_figure(fig, output_dir: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")


def available_models(root: Path) -> list[dict]:
    candidates = [
        {
            "name": "Qwen2.5-7B",
            "summary": None,
            "behavior": root
            / "outputs/behavior/qwen2_5_7b_lexical_control.metrics.json",
            "residual": root
            / "outputs/probes/qwen2_5_7b_lexical_control/residual_probe_metrics.json",
            "mlp": root
            / "outputs/probes/qwen2_5_7b_lexical_control_mlp/mlp_probe_metrics.json",
            "patching": root
            / "outputs/patching/controlled_comparison/patching_control_comparison.csv",
        },
        {
            "name": "Qwen2.5-Coder-7B",
            "summary": root / "outputs/qwen2_5_coder_7b_summary.json",
            "behavior": root
            / "outputs/behavior/qwen2_5_coder_7b_lexical_control.metrics.json",
            "residual": root
            / "outputs/probes/qwen2_5_coder_7b_lexical_control/residual_probe_metrics.json",
            "mlp": root
            / "outputs/probes/qwen2_5_coder_7b_lexical_control_mlp/mlp_probe_metrics.json",
            "patching": root
            / "outputs/patching/qwen2_5_coder_7b_controlled/patching_control_comparison.csv",
        },
        {
            "name": "Mistral-7B",
            "summary": root / "outputs/mistral_7b_instruct_v0_3_summary.json",
            "behavior": root
            / "outputs/behavior/mistral_7b_instruct_v0_3_lexical_control.metrics.json",
            "residual": root
            / "outputs/probes/mistral_7b_instruct_v0_3_lexical_control/residual_probe_metrics.json",
            "mlp": root
            / "outputs/probes/mistral_7b_instruct_v0_3_lexical_control_mlp/mlp_probe_metrics.json",
            "patching": root
            / "outputs/patching/mistral_7b_instruct_v0_3_controlled/patching_control_comparison.csv",
        },
    ]
    return [
        model
        for model in candidates
        if model["behavior"].exists()
        and model["residual"].exists()
        and model["mlp"].exists()
    ]


def write_model_table(models: list[dict], tables: Path) -> list[dict]:
    rows = []
    for model in models:
        behavior = load_json(model["behavior"])
        residual = load_json(model["residual"])
        mlp = load_json(model["mlp"])
        residual_best = max(residual, key=lambda row: row["macro_f1"])
        mlp_best = max(mlp, key=lambda row: row["macro_f1"])
        row = {
            "model": model["name"],
            "lexical_accuracy": behavior["accuracy"],
            "lexical_macro_f1": behavior["macro_f1"],
            "lexical_invalid_outputs": behavior["invalid_outputs"],
            "calculator_recall": behavior["per_label"]["calculator"]["recall"],
            "python_recall": behavior["per_label"]["python"]["recall"],
            "calculator_to_python": behavior["confusion_matrix"]["calculator"][
                "python"
            ],
            "residual_best_layer": residual_best["layer"],
            "residual_best_macro_f1": residual_best["macro_f1"],
            "mlp_best_layer": mlp_best["layer"],
            "mlp_best_macro_f1": mlp_best["macro_f1"],
        }
        if model["patching"].exists():
            with model["patching"].open(newline="", encoding="utf-8") as handle:
                patching = list(csv.DictReader(handle))
            best = max(patching, key=lambda item: float(item["treatment_mean_effect"]))
            row.update(
                {
                    "patching_best_layer": int(best["layer"]),
                    "patching_effect": float(best["treatment_mean_effect"]),
                    "patching_delta_python": float(best["delta_vs_python"]),
                    "patching_delta_none": float(best["delta_vs_none"]),
                }
            )
        rows.append(row)

    columns = sorted({key for row in rows for key in row})
    with (tables / "model_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_statistics(root: Path, tables: Path) -> list[dict]:
    tests = []
    for model, path in [
        (
            "Qwen2.5-7B",
            root
            / "outputs/patching/controlled_comparison/patching_control_comparison.csv",
        ),
        (
            "Qwen2.5-Coder-7B",
            root
            / "outputs/patching/qwen2_5_coder_7b_controlled/patching_control_comparison.csv",
        ),
        (
            "Mistral-7B",
            root
            / "outputs/patching/mistral_7b_instruct_v0_3_controlled/patching_control_comparison.csv",
        ),
    ]:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        best = max(rows, key=lambda row: float(row["treatment_mean_effect"]))
        for control in ("python", "none"):
            tests.append(
                {
                    "family": "controlled_patching",
                    "model": model,
                    "comparison": f"calculator_vs_{control}",
                    "layer": int(best["layer"]),
                    "effect_delta": float(best[f"delta_vs_{control}"]),
                    "raw_p": float(best[f"p_vs_{control}"]),
                }
            )

    if tests:
        adjusted = holm_adjust([row["raw_p"] for row in tests])
        for row, value in zip(tests, adjusted, strict=True):
            row["holm_p"] = value
            row["significant_0_05"] = value < 0.05
        with (tables / "statistical_tests.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(tests[0]))
            writer.writeheader()
            writer.writerows(tests)
    return tests


def plot_confusions(models: list[dict], figures: Path, plt) -> None:
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4))
    axes = np.atleast_1d(axes)
    for axis, model in zip(axes, models, strict=True):
        metrics = load_json(model["behavior"])
        matrix = np.asarray(
            [
                [
                    metrics["confusion_matrix"][actual].get(predicted, 0)
                    for predicted in LABELS
                ]
                for actual in LABELS
            ]
        )
        image = axis.imshow(matrix, cmap="Blues")
        for row in range(4):
            for column in range(4):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
        invalid = metrics["invalid_outputs"]
        axis.set_title(f"{model['name']}\ninvalid={invalid}")
        axis.set_xticks(range(4), LABELS, rotation=35, ha="right")
        axis.set_yticks(range(4), LABELS)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        fig.colorbar(image, ax=axis, fraction=0.046)
    save_figure(fig, figures, "behavior_confusion_matrices")
    plt.close(fig)


def plot_probes(models: list[dict], figures: Path, plt) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for model in models:
        color = COLORS[model["name"]]
        for axis, key, title in [
            (axes[0], "residual", "Residual stream"),
            (axes[1], "mlp", "MLP output"),
        ]:
            rows = load_json(model[key])
            axis.plot(
                [row["layer"] for row in rows],
                [row["macro_f1"] for row in rows],
                marker="o",
                markersize=2.5,
                label=model["name"],
                color=color,
            )
            axis.set_title(title)
            axis.set_xlabel("Layer")
            axis.set_ylim(0, 1.05)
            axis.axhline(0.25, color="gray", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Held-out lexical-control macro-F1")
    axes[1].legend(frameon=False)
    save_figure(fig, figures, "layerwise_probe_comparison")
    plt.close(fig)


def plot_patching(models: list[dict], figures: Path, plt) -> None:
    usable = [model for model in models if model["patching"].exists()]
    if not usable:
        return
    fig, axis = plt.subplots(figsize=(7, 4))
    for model in usable:
        with model["patching"].open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        axis.plot(
            [int(row["layer"]) for row in rows],
            [float(row["treatment_mean_effect"]) for row in rows],
            marker="o",
            markersize=3,
            label=model["name"],
            color=COLORS[model["name"]],
        )
    axis.axhline(0, color="black", linewidth=1)
    axis.set_xlabel("Patched layer")
    axis.set_ylabel("Calculator minus Python logit effect")
    axis.legend(frameon=False)
    save_figure(fig, figures, "controlled_patching_comparison")
    plt.close(fig)


def plot_component_ablations(root: Path, figures: Path, tables: Path, plt) -> None:
    candidates = [
        (
            "Qwen2.5-7B",
            root / "outputs/ablations/calculator_neurons_k20.json",
            root / "outputs/ablations/calculator_heads_k20.json",
        ),
        (
            "Qwen2.5-Coder-7B",
            root / "outputs/ablations/qwen2_5_coder_7b_calculator_neurons_k20.json",
            root / "outputs/ablations/qwen2_5_coder_7b_calculator_heads_k20.json",
        ),
        (
            "Mistral-7B",
            root
            / "outputs/ablations/mistral_7b_instruct_v0_3_calculator_neurons_k20.json",
            root
            / "outputs/ablations/mistral_7b_instruct_v0_3_calculator_heads_k20.json",
        ),
    ]
    rows = []
    for model, neuron_path, head_path in candidates:
        if not neuron_path.exists() or not head_path.exists():
            continue
        for component, path in [("Neurons", neuron_path), ("Heads", head_path)]:
            result = load_json(path)
            rows.append(
                {
                    "model": model,
                    "component": component,
                    "mean_effect": result["summary"]["top_ranked"]["mean_effect"],
                    "ci_low": result["summary"]["top_ranked"]["ci95"][0],
                    "ci_high": result["summary"]["top_ranked"]["ci95"][1],
                    "random_control": result["summary"]["random_control"][
                        "mean_effect"
                    ],
                    "p": result["summary"]["paired_permutation_p"],
                    "flip_rate": result["summary"]["top_flip_rate"],
                }
            )
    if not rows:
        return
    with (tables / "component_ablation_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    models = list(dict.fromkeys(row["model"] for row in rows))
    x = np.arange(len(models))
    width = 0.35
    fig, axis = plt.subplots(figsize=(7, 4))
    for offset, component in enumerate(("Neurons", "Heads")):
        component_rows = {
            row["model"]: row for row in rows if row["component"] == component
        }
        effects = [component_rows[model]["mean_effect"] for model in models]
        low = [component_rows[model]["ci_low"] for model in models]
        high = [component_rows[model]["ci_high"] for model in models]
        errors = [
            [effect - lower for effect, lower in zip(effects, low, strict=True)],
            [upper - effect for effect, upper in zip(effects, high, strict=True)],
        ]
        axis.bar(
            x + (offset - 0.5) * width,
            effects,
            width,
            yerr=errors,
            capsize=3,
            label=component,
        )
    axis.axhline(0, color="black", linewidth=1)
    axis.set_xticks(x, models, rotation=15)
    axis.set_ylabel("Ablation effect on calculator margin")
    axis.legend(frameon=False)
    save_figure(fig, figures, "component_ablation_comparison")
    plt.close(fig)


def plot_main_model(root: Path, figures: Path, tables: Path, plt) -> None:
    steering_paths = [
        (
            "Calculator direction",
            root / "outputs/patching/steering_calculator_layer26.json",
            "#3366cc",
        ),
        (
            "Negative direction",
            root / "outputs/patching/steering_negative_calculator_layer26.json",
            "#dc3912",
        ),
        (
            "Python control",
            root / "outputs/patching/steering_python_control_layer26.json",
            "#109618",
        ),
        (
            "Calculator high dose",
            root / "outputs/patching/steering_calculator_highdose_layer26.json",
            "#990099",
        ),
        (
            "Negative high dose",
            root
            / "outputs/patching/steering_negative_calculator_highdose_layer26.json",
            "#ff9900",
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, path, color in steering_paths:
        if not path.exists():
            continue
        rows = load_json(path)["summary"]
        axes[0].plot(
            [row["alpha"] for row in rows],
            [row["mean_effect"] for row in rows],
            marker="o",
            label=name,
            color=color,
        )
        axes[1].plot(
            [row["alpha"] for row in rows],
            [row["flip_rate"] for row in rows],
            marker="o",
            label=name,
            color=color,
        )
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set(xlabel="Steering alpha", ylabel="Mean logit effect")
    axes[1].set(xlabel="Steering alpha", ylabel="Behavioral flip rate", ylim=(-0.03, 1.03))
    axes[1].legend(frameon=False, fontsize=8)
    save_figure(fig, figures, "steering_dose_response")
    plt.close(fig)

    validation = root / "outputs/ablations/l26h4_validation.json"
    challenge = root / "outputs/ablations/l26h4_challenge.json"
    if validation.exists() and challenge.exists():
        fig, axis = plt.subplots(figsize=(7, 4))
        x = np.arange(4)
        width = 0.35
        for offset, (name, path) in enumerate(
            [("Validation", validation), ("Challenge", challenge)]
        ):
            result = load_json(path)
            effects = [
                result["results"][label]["target"]["mean_effect"] for label in LABELS
            ]
            axis.bar(x + (offset - 0.5) * width, effects, width, label=name)
        axis.axhline(0, color="black", linewidth=1)
        axis.set_xticks(x, LABELS, rotation=20)
        axis.set_ylabel("L26H4 ablation effect on own-label margin")
        axis.legend(frameon=False)
        save_figure(fig, figures, "l26h4_specificity")
        plt.close(fig)

    attention = root / "outputs/ablations/l26h4_attention.json"
    if attention.exists():
        token_masses: dict[str, float] = {}
        for row in load_json(attention)["summary"]["calculator"]["top_tokens"]:
            token = row["token"].replace("\u010a", "\\n")
            token_masses[token] = token_masses.get(token, 0.0) + row["attention_mass"]
        rows = sorted(token_masses.items(), key=lambda item: item[1], reverse=True)[:10]
        tokens = [token for token, _mass in rows][::-1]
        masses = [mass for _token, mass in rows][::-1]
        fig, axis = plt.subplots(figsize=(7, 4))
        axis.barh(tokens, masses, color="#3366cc")
        axis.set_xlabel("Aggregated attention mass")
        axis.set_title("L26H4 attention on calculator prompts")
        save_figure(fig, figures, "l26h4_attention_tokens")
        plt.close(fig)

    seed_paths = [
        (
            "patching",
            root / "outputs/patching/seeds/summary.json",
            "treatment_mean_effect",
        ),
        (
            "L26H4",
            root / "outputs/ablations/l26h4_seeds/summary.json",
            "calculator_mean_effect",
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plotted = False
    seed_table_rows = []
    for axis, (name, path, key) in zip(axes, seed_paths, strict=True):
        if not path.exists():
            continue
        rows = load_json(path)["seeds"]
        for row in rows:
            seed_table_rows.append(
                {
                    "experiment": name,
                    "seed": row["seed"],
                    "effect": row[key],
                    "passed": row["passed"],
                }
            )
        axis.bar([str(row["seed"]) for row in rows], [row[key] for row in rows])
        axis.axhline(0, color="black", linewidth=1)
        axis.set(xlabel="Seed", ylabel="Mean effect", title=name)
        plotted = True
    if plotted:
        save_figure(fig, figures, "multiseed_replication")
    plt.close(fig)
    if seed_table_rows:
        with (tables / "multiseed_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["experiment", "seed", "effect", "passed"],
            )
            writer.writeheader()
            writer.writerows(seed_table_rows)

    order_path = root / "outputs/ablations/l26h4_prompt_order.json"
    if order_path.exists():
        rows = load_json(order_path)["orders"]
        with (tables / "l26h4_prompt_order.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "order",
                    "calculator_position",
                    "samples",
                    "mean_effect",
                    "flip_rate",
                ],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "order": " > ".join(row["order"]),
                        "calculator_position": row["calculator_position"],
                        "samples": row["samples"],
                        "mean_effect": row["mean_effect"],
                        "flip_rate": row["flip_rate"],
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create local statistical tables and paper-ready figures"
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--figure-dir", type=Path, default=Path("paper/figures"))
    parser.add_argument("--table-dir", type=Path, default=Path("paper/tables"))
    parser.add_argument(
        "--summary", type=Path, default=Path("paper/local_report.summary.json")
    )
    args = parser.parse_args()

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Install research dependencies with matplotlib.") from exc

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    models = available_models(args.root)
    if not models:
        raise SystemExit("No complete local model outputs were found.")

    model_rows = write_model_table(models, args.table_dir)
    tests = write_statistics(args.root, args.table_dir)
    plot_confusions(models, args.figure_dir, plt)
    plot_probes(models, args.figure_dir, plt)
    plot_patching(models, args.figure_dir, plt)
    plot_component_ablations(args.root, args.figure_dir, args.table_dir, plt)
    plot_main_model(args.root, args.figure_dir, args.table_dir, plt)

    summary = {
        "models_included": [model["name"] for model in models],
        "model_table_rows": len(model_rows),
        "statistical_tests": len(tests),
        "figures": sorted(path.name for path in args.figure_dir.glob("*.png")),
        "tables": sorted(path.name for path in args.table_dir.glob("*.csv")),
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
