#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Causally screen individual heads for tool-specific effects"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tool", default="calculator")
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--samples-per-label", type=int, default=8)
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

    ranking_data = json.loads(args.rankings.read_text(encoding="utf-8"))
    ranked = ranking_data["heads"][args.tool]
    positive = [row for row in ranked if row["effect_size"] > 0]
    negative = [row for row in ranked if row["effect_size"] < 0]
    candidates = (positive[: args.candidates] + negative[: args.candidates])

    all_rows = list(read_jsonl(args.input))
    rows_by_label = {
        label: [row for row in all_rows if row["label"] == label][
            : args.samples_per_label
        ]
        for label in LABELS
    }

    def tokenize(row):
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": format_tool_prompt(row["query"])}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return tokenizer(rendered, return_tensors="pt").to(model.device)

    def label_margin(logits, label):
        final = logits[0, -1]
        own = label_ids[label]
        competitors = [token for name, token in label_ids.items() if name != label]
        return float((final[own] - final[competitors].max()).item())

    cached = {}
    for label, rows in rows_by_label.items():
        cached[label] = []
        for row in rows:
            inputs = tokenize(row)
            with torch.inference_mode():
                baseline = label_margin(model(**inputs, use_cache=False).logits, label)
            if baseline > 0:
                cached[label].append((row, inputs, baseline))
    if any(not rows for rows in cached.values()):
        raise SystemExit(
            "At least one class has no correctly classified calibration prompts."
        )

    results = []
    for index, candidate in enumerate(candidates, start=1):
        layer_index = candidate["layer"]
        head = candidate["head"]
        module = resolve_attention_output_projection(layers[layer_index])
        effects = {}

        def ablate(_module, hook_inputs):
            value = hook_inputs[0].clone()
            start, end = head * head_dim, (head + 1) * head_dim
            value[:, :, start:end] = 0
            return (value, *hook_inputs[1:])

        for label, rows in cached.items():
            label_effects = []
            for _row, inputs, baseline in rows:
                handle = module.register_forward_pre_hook(ablate)
                try:
                    with torch.inference_mode():
                        changed = label_margin(
                            model(**inputs, use_cache=False).logits, label
                        )
                finally:
                    handle.remove()
                label_effects.append(changed - baseline)
            effects[label] = {
                "mean": float(np.mean(label_effects)),
                "std": float(np.std(label_effects)),
                "samples": len(label_effects),
            }

        calculator_drop = -effects[args.tool]["mean"]
        unrelated_drop = max(
            0.0,
            max(-effects[label]["mean"] for label in LABELS if label != args.tool),
        )
        results.append(
            {
                **candidate,
                "effects": effects,
                "calculator_drop": calculator_drop,
                "largest_unrelated_drop": unrelated_drop,
                "specificity_score": calculator_drop - unrelated_drop,
            }
        )
        print(f"Screened {index}/{len(candidates)}")

    results.sort(key=lambda row: row["specificity_score"], reverse=True)
    payload = {
        "model": args.model,
        "tool": args.tool,
        "candidate_count": len(candidates),
        "samples_per_label": args.samples_per_label,
        "ranking": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Top calculator-specific candidates:")
    for row in results[:10]:
        print(
            row["layer"],
            row["head"],
            round(row["specificity_score"], 4),
            {label: round(value["mean"], 4) for label, value in row["effects"].items()},
        )


if __name__ == "__main__":
    main()
