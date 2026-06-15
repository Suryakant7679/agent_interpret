#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from tool_circuits.hf_utils import resolve_decoder_layers
from tool_circuits.io import read_jsonl
from tool_circuits.patching import select_patching_rows
from tool_circuits.prompting import format_tool_prompt
from tool_circuits.statistics import bootstrap_mean_interval


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch final-token residual states from clean prompts into errors"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--source-input", type=Path, required=True)
    parser.add_argument("--target-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--positive-label", default="calculator")
    parser.add_argument("--negative-label", default="python")
    parser.add_argument(
        "--source-label",
        help="Source class for treatment or matched control; defaults to positive-label.",
    )
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args()
    source_label = args.source_label or args.positive_label

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Install research dependencies: python3 -m pip install -e '.[research]'"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    positive_ids = tokenizer.encode(args.positive_label, add_special_tokens=False)
    negative_ids = tokenizer.encode(args.negative_label, add_special_tokens=False)
    if not positive_ids or not negative_ids:
        raise SystemExit(
            "Labels must tokenize to at least one token. "
            f"{args.positive_label}={positive_ids}, {args.negative_label}={negative_ids}"
        )
    positive_id = positive_ids[0]
    negative_id = negative_ids[0]
    if positive_id == negative_id:
        raise SystemExit(
            "Positive and negative labels share the same first token, so a "
            "first-token prefix-logit contrast is not identifiable."
        )
    if len(positive_ids) > 1 or len(negative_ids) > 1:
        print(
            "Using first-token prefix logits for multi-token labels: "
            f"{args.positive_label}={positive_ids}, "
            f"{args.negative_label}={negative_ids}"
        )
    source_ids = tokenizer.encode(source_label, add_special_tokens=False)
    if not source_ids:
        raise SystemExit(f"Source label tokenized to an empty sequence: {source_label}")
    source_id = source_ids[0]

    model_kwargs = {"torch_dtype": "auto", "device_map": args.device_map}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()
    layers = resolve_decoder_layers(model)

    def tokenize(row: dict):
        prompt = format_tool_prompt(row["query"])
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return tokenizer(rendered, return_tensors="pt").to(model.device)

    def logit_difference(logits) -> float:
        final_logits = logits[0, -1]
        return float((final_logits[positive_id] - final_logits[negative_id]).item())

    def baseline(row: dict, hidden_states: bool = False):
        with torch.inference_mode():
            return model(
                **tokenize(row),
                output_hidden_states=hidden_states,
                use_cache=False,
            )

    source_rows, target_rows = select_patching_rows(
        read_jsonl(args.source_input),
        read_jsonl(args.target_input),
        source_label=source_label,
        positive_label=args.positive_label,
        negative_label=args.negative_label,
        seed=args.seed,
    )

    clean_sources = []
    for row in source_rows:
        output = baseline(row, hidden_states=True)
        difference = logit_difference(output.logits)
        source_logit = output.logits[0, -1, source_id]
        contrast_id = negative_id if source_label == args.positive_label else positive_id
        source_preferred = float(
            (source_logit - output.logits[0, -1, contrast_id]).item()
        ) > 0
        if source_preferred:
            clean_sources.append(
                {
                    "row": row,
                    "difference": difference,
                    "states": [
                        state[:, -1, :].detach()
                        for state in output.hidden_states[1:]
                    ],
                }
            )
        if len(clean_sources) >= args.pairs:
            break

    error_targets = []
    for row in target_rows:
        output = baseline(row)
        difference = logit_difference(output.logits)
        if difference < 0:
            error_targets.append({"row": row, "difference": difference})
        if len(error_targets) >= args.pairs:
            break

    pair_count = min(len(clean_sources), len(error_targets), args.pairs)
    if pair_count == 0:
        raise SystemExit(
            "No usable pairs found. Sources must prefer the positive label and "
            "targets must prefer the negative label."
        )
    print(
        f"Using {pair_count} pairs: clean {source_label} sources and "
        f"{args.positive_label}->{args.negative_label} target errors. "
        f"Source class: {source_label}."
    )

    pair_results = []
    effects = np.zeros((pair_count, len(layers)), dtype=np.float64)
    recoveries = np.zeros_like(effects)
    flips = np.zeros_like(effects, dtype=np.int64)

    for pair_index in range(pair_count):
        source = clean_sources[pair_index]
        target = error_targets[pair_index]
        target_inputs = tokenize(target["row"])
        layer_rows = []
        denominator = source["difference"] - target["difference"]

        for layer_index, layer in enumerate(layers):
            source_state = source["states"][layer_index]

            def patch(_module, _inputs, output):
                if isinstance(output, tuple):
                    hidden = output[0].clone()
                    hidden[:, -1, :] = source_state.to(
                        device=hidden.device, dtype=hidden.dtype
                    )
                    return (hidden, *output[1:])
                hidden = output.clone()
                hidden[:, -1, :] = source_state.to(
                    device=hidden.device, dtype=hidden.dtype
                )
                return hidden

            handle = layer.register_forward_hook(patch)
            try:
                with torch.inference_mode():
                    patched_output = model(**target_inputs, use_cache=False)
            finally:
                handle.remove()

            patched_difference = logit_difference(patched_output.logits)
            effect = patched_difference - target["difference"]
            recovery = effect / denominator if abs(denominator) > 1e-12 else 0.0
            flipped = patched_difference > 0
            effects[pair_index, layer_index] = effect
            recoveries[pair_index, layer_index] = recovery
            flips[pair_index, layer_index] = int(flipped)
            layer_rows.append(
                {
                    "layer": layer_index,
                    "patched_logit_difference": patched_difference,
                    "effect": effect,
                    "recovery": recovery,
                    "flipped_to_positive": flipped,
                }
            )

        pair_results.append(
            {
                "source_id": source["row"]["id"],
                "source_query": source["row"]["query"],
                "source_logit_difference": source["difference"],
                "target_id": target["row"]["id"],
                "target_query": target["row"]["query"],
                "target_logit_difference": target["difference"],
                "layers": layer_rows,
            }
        )
        print(f"Completed pair {pair_index + 1}/{pair_count}")

    summary = []
    for layer_index in range(len(layers)):
        effect_mean, effect_low, effect_high = bootstrap_mean_interval(
            effects[:, layer_index], seed=args.seed
        )
        recovery_mean, recovery_low, recovery_high = bootstrap_mean_interval(
            recoveries[:, layer_index], seed=args.seed
        )
        summary.append(
            {
                "layer": layer_index,
                "mean_effect": effect_mean,
                "effect_ci95": [effect_low, effect_high],
                "mean_recovery": recovery_mean,
                "recovery_ci95": [recovery_low, recovery_high],
                "flip_rate": float(flips[:, layer_index].mean()),
            }
        )

    result = {
        "model": args.model,
        "positive_label": args.positive_label,
        "negative_label": args.negative_label,
        "source_label": source_label,
        "seed": args.seed,
        "pair_count": pair_count,
        "token_ids": {
            args.positive_label: positive_ids,
            args.negative_label: negative_ids,
            source_label: source_ids,
        },
        "metric": "positive_label_first_token_prefix_minus_negative_label_first_token_prefix",
        "summary": summary,
        "pairs": pair_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    best = max(summary, key=lambda row: row["mean_effect"])
    print(json.dumps({"best_layer": best, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
