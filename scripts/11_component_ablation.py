#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from tool_circuits.hf_utils import (
    resolve_attention_output_projection,
    resolve_decoder_layers,
    resolve_down_projection,
)
from tool_circuits import LABELS
from tool_circuits.io import read_jsonl
from tool_circuits.prompting import format_tool_prompt
from tool_circuits.statistics import (
    bootstrap_mean_interval,
    paired_permutation_pvalue,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablate ranked neurons or heads")
    parser.add_argument("--model", required=True)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--component", choices=["neurons", "heads"], required=True)
    parser.add_argument("--tool", default="calculator")
    parser.add_argument("--contrast", default="python")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--effect-sign",
        choices=["absolute", "positive", "negative"],
        default="absolute",
    )
    parser.add_argument("--random-trials", type=int, default=10)
    parser.add_argument(
        "--evaluation-label",
        help="Prompt class to evaluate; defaults to --tool.",
    )
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install research dependencies first.") from exc

    rng = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    label_token_sequences = {
        label: tokenizer.encode(label, add_special_tokens=False) for label in LABELS
    }
    if any(not ids for ids in label_token_sequences.values()):
        raise SystemExit("One or more tool labels tokenized to an empty sequence.")
    label_token_ids = {
        label: ids[0] for label, ids in label_token_sequences.items()
    }
    if len(set(label_token_ids.values())) != len(label_token_ids):
        raise SystemExit(
            "Tool labels do not have distinct first-token prefixes for this tokenizer."
        )

    kwargs = {"torch_dtype": "auto", "device_map": "auto"}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    model.eval()
    layers = resolve_decoder_layers(model)
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads

    rankings = json.loads(args.rankings.read_text(encoding="utf-8"))
    candidates = rankings[args.component][args.tool]
    if args.effect_sign == "positive":
        candidates = [row for row in candidates if row["effect_size"] > 0]
    elif args.effect_sign == "negative":
        candidates = [row for row in candidates if row["effect_size"] < 0]
    selected = candidates[: args.top_k]
    if len(selected) < args.top_k:
        raise SystemExit(
            f"Only {len(selected)} components match effect sign {args.effect_sign!r}; "
            f"requested {args.top_k}."
        )
    grouped: dict[int, list[int]] = {}
    key = "neuron" if args.component == "neurons" else "head"
    for row in selected:
        grouped.setdefault(row["layer"], []).append(row[key])

    random_groups = []
    for _trial in range(args.random_trials):
        trial_groups: dict[int, list[int]] = {}
        for layer, indices in grouped.items():
            width = (
                resolve_down_projection(layers[layer]).in_features
                if args.component == "neurons"
                else num_heads
            )
            pool = [index for index in range(width) if index not in indices]
            trial_groups[layer] = rng.sample(pool, len(indices))
        random_groups.append(trial_groups)

    evaluation_label = args.evaluation_label or args.tool
    evaluation_id = label_token_ids[evaluation_label]
    competitor_ids = [
        token_id
        for label, token_id in label_token_ids.items()
        if label != evaluation_label
    ]
    candidate_rows = [
        row for row in read_jsonl(args.input) if row["label"] == evaluation_label
    ]

    def tokenize(row):
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": format_tool_prompt(row["query"])}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return tokenizer(rendered, return_tensors="pt").to(model.device)

    def difference(logits):
        final = logits[0, -1]
        strongest_competitor = final[competitor_ids].max()
        return float((final[evaluation_id] - strongest_competitor).item())

    def run_ablation(inputs, groups):
        handles = []
        for layer_index, indices in groups.items():
            module = (
                resolve_down_projection(layers[layer_index])
                if args.component == "neurons"
                else resolve_attention_output_projection(layers[layer_index])
            )

            def ablate(_module, hook_inputs, chosen=tuple(indices)):
                value = hook_inputs[0].clone()
                if args.component == "neurons":
                    value[:, :, list(chosen)] = 0
                else:
                    for head in chosen:
                        start, end = head * head_dim, (head + 1) * head_dim
                        value[:, :, start:end] = 0
                return (value, *hook_inputs[1:])

            handles.append(module.register_forward_pre_hook(ablate))
        try:
            with torch.inference_mode():
                return difference(model(**inputs, use_cache=False).logits)
        finally:
            for handle in handles:
                handle.remove()

    examples = []
    for row in candidate_rows:
        inputs = tokenize(row)
        with torch.inference_mode():
            baseline = difference(model(**inputs, use_cache=False).logits)
        if baseline <= 0:
            continue
        top = run_ablation(inputs, grouped)
        controls = [run_ablation(inputs, groups) for groups in random_groups]
        control = float(np.mean(controls))
        examples.append(
            {
                "id": row["id"],
                "baseline": baseline,
                "top_ranked": top,
                "random_control": control,
                "random_trials": controls,
                "top_effect": top - baseline,
                "control_effect": control - baseline,
            }
        )
        print(f"Completed {len(examples)}/{args.limit}")
        if len(examples) >= args.limit:
            break
    if not examples:
        raise SystemExit("No correctly classified positive-tool prompts were found.")

    top_effect = np.asarray([row["top_effect"] for row in examples])
    control_effect = np.asarray([row["control_effect"] for row in examples])
    top_mean, top_low, top_high = bootstrap_mean_interval(top_effect, seed=args.seed)
    control_mean, control_low, control_high = bootstrap_mean_interval(
        control_effect, seed=args.seed
    )
    summary = {
        "top_ranked": {"mean_effect": top_mean, "ci95": [top_low, top_high]},
        "random_control": {
            "mean_effect": control_mean,
            "ci95": [control_low, control_high],
        },
        "paired_permutation_p": paired_permutation_pvalue(
            top_effect, control_effect, seed=args.seed
        ),
        "top_flip_rate": float(np.mean([row["top_ranked"] < 0 for row in examples])),
        "control_flip_rate": float(
            np.mean([row["random_control"] < 0 for row in examples])
        ),
    }
    payload = {
        "model": args.model,
        "component": args.component,
        "tool": args.tool,
        "contrast": args.contrast,
        "evaluation_label": evaluation_label,
        "metric": "evaluation_label_first_token_minus_strongest_competing_label_first_token",
        "label_token_ids": label_token_ids,
        "label_token_sequences": label_token_sequences,
        "effect_sign": args.effect_sign,
        "top_k": args.top_k,
        "samples": len(examples),
        "selected": selected,
        "random_matched_trials": random_groups,
        "summary": summary,
        "examples": examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
