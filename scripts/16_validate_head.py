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

from tool_circuits import LABELS
from tool_circuits.hf_utils import (
    resolve_attention_output_projection,
    resolve_decoder_layers,
)
from tool_circuits.io import read_jsonl
from tool_circuits.prompting import format_tool_prompt
from tool_circuits.statistics import (
    bootstrap_mean_interval,
    paired_permutation_pvalue,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Confirm one candidate head on held-out prompts and matched controls"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--head", type=int, required=True)
    parser.add_argument("--samples-per-label", type=int, default=32)
    parser.add_argument("--random-heads", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--shuffle-prompts",
        action="store_true",
        help="Shuffle each label's prompt pool with --seed before sampling.",
    )
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install research dependencies first.") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    label_ids = {
        label: tokenizer.encode(label, add_special_tokens=False)[0] for label in LABELS
    }
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
    if not 0 <= args.layer < len(layers) or not 0 <= args.head < num_heads:
        raise SystemExit("Layer or head index is out of range.")

    rng = random.Random(args.seed)
    random_heads = rng.sample(
        [head for head in range(num_heads) if head != args.head],
        min(args.random_heads, num_heads - 1),
    )
    module = resolve_attention_output_projection(layers[args.layer])

    def tokenize(row):
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": format_tool_prompt(row["query"])}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return tokenizer(rendered, return_tensors="pt").to(model.device)

    def margin(logits, label):
        final = logits[0, -1]
        own = label_ids[label]
        competitors = [token for name, token in label_ids.items() if name != label]
        return float((final[own] - final[competitors].max()).item())

    def ablated_margin(inputs, label, head):
        def ablate(_module, hook_inputs):
            value = hook_inputs[0].clone()
            start, end = head * head_dim, (head + 1) * head_dim
            value[:, :, start:end] = 0
            return (value, *hook_inputs[1:])

        handle = module.register_forward_pre_hook(ablate)
        try:
            with torch.inference_mode():
                return margin(model(**inputs, use_cache=False).logits, label)
        finally:
            handle.remove()

    all_rows = list(read_jsonl(args.input))
    results = {}
    for label in LABELS:
        label_rows = [row for row in all_rows if row["label"] == label]
        if args.shuffle_prompts:
            rng.shuffle(label_rows)
        examples = []
        for row in label_rows:
            inputs = tokenize(row)
            with torch.inference_mode():
                baseline = margin(model(**inputs, use_cache=False).logits, label)
            if baseline <= 0:
                continue
            target = ablated_margin(inputs, label, args.head)
            controls = [
                ablated_margin(inputs, label, random_head) for random_head in random_heads
            ]
            examples.append(
                {
                    "id": row["id"],
                    "baseline_margin": baseline,
                    "target_margin": target,
                    "target_effect": target - baseline,
                    "random_head_margins": controls,
                    "random_control_effect": float(np.mean(controls) - baseline),
                    "target_flip": target < 0,
                }
            )
            if len(examples) >= args.samples_per_label:
                break
        if len(examples) < args.samples_per_label:
            print(
                f"Warning: {label} has {len(examples)} correctly classified prompts; "
                f"requested {args.samples_per_label}."
            )
        target_effect = np.asarray([row["target_effect"] for row in examples])
        control_effect = np.asarray(
            [row["random_control_effect"] for row in examples]
        )
        target_mean, target_low, target_high = bootstrap_mean_interval(
            target_effect, seed=args.seed
        )
        control_mean, control_low, control_high = bootstrap_mean_interval(
            control_effect, seed=args.seed
        )
        results[label] = {
            "samples": len(examples),
            "target": {
                "mean_effect": target_mean,
                "ci95": [target_low, target_high],
                "flip_rate": float(np.mean([row["target_flip"] for row in examples])),
            },
            "random_control": {
                "mean_effect": control_mean,
                "ci95": [control_low, control_high],
            },
            "paired_permutation_p": paired_permutation_pvalue(
                target_effect, control_effect, seed=args.seed
            ),
            "examples": examples,
        }
        print(f"Completed {label}: {len(examples)} prompts")

    calculator_drop = -results["calculator"]["target"]["mean_effect"]
    unrelated_drop = max(
        0.0,
        max(
            -results[label]["target"]["mean_effect"]
            for label in LABELS
            if label != "calculator"
        ),
    )
    payload = {
        "model": args.model,
        "input": str(args.input),
        "layer": args.layer,
        "head": args.head,
        "random_heads": random_heads,
        "samples_per_label": args.samples_per_label,
        "seed": args.seed,
        "shuffle_prompts": args.shuffle_prompts,
        "calculator_drop": calculator_drop,
        "largest_unrelated_drop": unrelated_drop,
        "specificity_score": calculator_drop - unrelated_drop,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "layer": args.layer,
                "head": args.head,
                "specificity_score": payload["specificity_score"],
                "effects": {
                    label: results[label]["target"]["mean_effect"] for label in LABELS
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
