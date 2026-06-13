#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
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
from tool_circuits.statistics import bootstrap_mean_interval


DESCRIPTIONS = {
    "web_search": "use for current, recent, or externally verifiable information.",
    "calculator": "use for exact arithmetic.",
    "python": "use for code execution, data analysis, or simulation.",
    "none": "use when no tool is needed.",
}


def format_variant(query: str, order: tuple[str, ...]) -> str:
    tools = "\n".join(
        f"{index}. {label}: {DESCRIPTIONS[label]}"
        for index, label in enumerate(order, start=1)
    )
    labels = ", ".join(order)
    return f"""You are an AI assistant with access to tools.

Available tools:
{tools}

User query:
{query.strip()}

Which tool should be used?
Answer with exactly one label:
{labels}."""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate head effect under prompt tool-order permutations"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=26)
    parser.add_argument("--head", type=int, default=4)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--orders", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
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
    module = resolve_attention_output_projection(layers[args.layer])

    rng = random.Random(args.seed)
    all_orders = list(itertools.permutations(LABELS))
    rng.shuffle(all_orders)
    orders = [tuple(LABELS)] + [
        order for order in all_orders if order != tuple(LABELS)
    ][: max(0, args.orders - 1)]
    rows = [row for row in read_jsonl(args.input) if row["label"] == "calculator"]

    def margin(logits):
        final = logits[0, -1]
        own = label_ids["calculator"]
        others = [token for label, token in label_ids.items() if label != "calculator"]
        return float((final[own] - final[others].max()).item())

    def ablate(_module, hook_inputs):
        value = hook_inputs[0].clone()
        start, end = args.head * head_dim, (args.head + 1) * head_dim
        value[:, :, start:end] = 0
        return (value, *hook_inputs[1:])

    order_results = []
    for order_index, order in enumerate(orders, start=1):
        examples = []
        for row in rows:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": format_variant(row["query"], order)}],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                baseline = margin(model(**inputs, use_cache=False).logits)
            if baseline <= 0:
                continue
            handle = module.register_forward_pre_hook(ablate)
            try:
                with torch.inference_mode():
                    changed = margin(model(**inputs, use_cache=False).logits)
            finally:
                handle.remove()
            examples.append(
                {
                    "id": row["id"],
                    "baseline": baseline,
                    "ablated": changed,
                    "effect": changed - baseline,
                    "flipped": changed < 0,
                }
            )
            if len(examples) >= args.samples:
                break
        effects = np.asarray([row["effect"] for row in examples])
        if not len(effects):
            order_results.append(
                {
                    "order": list(order),
                    "calculator_position": order.index("calculator"),
                    "samples": 0,
                    "mean_effect": None,
                    "ci95": None,
                    "flip_rate": None,
                    "examples": [],
                }
            )
            print(f"No correctly classified calculator prompts for order {order}")
            continue
        mean, low, high = bootstrap_mean_interval(effects, seed=args.seed)
        order_results.append(
            {
                "order": list(order),
                "calculator_position": order.index("calculator"),
                "samples": len(examples),
                "mean_effect": mean,
                "ci95": [low, high],
                "flip_rate": float(np.mean([row["flipped"] for row in examples])),
                "examples": examples,
            }
        )
        print(f"Completed order {order_index}/{len(orders)}: {order}")

    payload = {
        "model": args.model,
        "layer": args.layer,
        "head": args.head,
        "orders": order_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for result in order_results:
        print(
            result["order"],
            result["samples"],
            None if result["mean_effect"] is None else round(result["mean_effect"], 4),
            None if result["flip_rate"] is None else round(result["flip_rate"], 4),
        )


if __name__ == "__main__":
    main()
