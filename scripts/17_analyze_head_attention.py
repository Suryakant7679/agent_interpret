#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from tool_circuits import LABELS
from tool_circuits.io import read_jsonl
from tool_circuits.prompting import format_tool_prompt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze which prompt tokens a candidate attention head reads"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--head", type=int, required=True)
    parser.add_argument("--samples-per-label", type=int, default=16)
    parser.add_argument("--top-tokens", type=int, default=20)
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install research dependencies first.") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    kwargs = {
        "torch_dtype": "auto",
        "device_map": "auto",
        "attn_implementation": "eager",
    }
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
    if not 0 <= args.layer < model.config.num_hidden_layers:
        raise SystemExit("Layer is out of range.")
    if not 0 <= args.head < model.config.num_attention_heads:
        raise SystemExit("Head is out of range.")

    rows = list(read_jsonl(args.input))
    selected = {
        label: [row for row in rows if row["label"] == label][
            : args.samples_per_label
        ]
        for label in LABELS
    }
    token_mass: dict[str, defaultdict[str, float]] = {
        label: defaultdict(float) for label in LABELS
    }
    role_mass: dict[str, defaultdict[str, float]] = {
        label: defaultdict(float) for label in LABELS
    }
    position_mass: dict[str, defaultdict[int, float]] = {
        label: defaultdict(float) for label in LABELS
    }
    examples = []

    for label, label_rows in selected.items():
        for row in label_rows:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": format_tool_prompt(row["query"])}],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                output = model(**inputs, output_attentions=True, use_cache=False)
            if output.attentions is None:
                raise SystemExit(
                    "The installed Transformers/model implementation did not return "
                    "attention weights. Upgrade Transformers or use an eager-attention "
                    "compatible model build."
                )
            attention = (
                output.attentions[args.layer][0, args.head, -1]
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            calculator_positions = [
                index
                for index, token in enumerate(tokens)
                if tokenizer.decode([inputs["input_ids"][0, index]]).strip()
                == "calculator"
            ]
            top_indices = np.argsort(attention)[::-1][: args.top_tokens]
            top = []
            for index in top_indices:
                token = tokens[int(index)]
                weight = float(attention[int(index)])
                normalized = token.replace("Ġ", " ").replace("▁", " ").strip()
                normalized = normalized or token
                token_mass[label][normalized] += weight
                position_mass[label][len(tokens) - 1 - int(index)] += weight
                if int(index) in calculator_positions:
                    calculator_rank = calculator_positions.index(int(index))
                    if calculator_rank == 0:
                        role_mass[label]["calculator_tool_description"] += weight
                    elif calculator_rank == len(calculator_positions) - 1:
                        role_mass[label]["calculator_allowed_label_list"] += weight
                    else:
                        role_mass[label]["calculator_other"] += weight
                top.append(
                    {
                        "position": int(index),
                        "distance_from_decision": len(tokens) - 1 - int(index),
                        "token": token,
                        "decoded": tokenizer.decode([inputs["input_ids"][0, index]]),
                        "weight": weight,
                    }
                )
            examples.append(
                {
                    "id": row["id"],
                    "label": label,
                    "query": row["query"],
                    "top_attention": top,
                }
            )
            print(f"Completed {len(examples)}/{sum(map(len, selected.values()))}")

    summary = {}
    for label in LABELS:
        token_rows = sorted(
            token_mass[label].items(), key=lambda item: item[1], reverse=True
        )[: args.top_tokens]
        position_rows = sorted(
            position_mass[label].items(), key=lambda item: item[1], reverse=True
        )[: args.top_tokens]
        summary[label] = {
            "top_tokens": [
                {"token": token, "attention_mass": mass}
                for token, mass in token_rows
            ],
            "top_distances": [
                {"distance_from_decision": distance, "attention_mass": mass}
                for distance, mass in position_rows
            ],
            "semantic_roles": dict(role_mass[label]),
        }

    payload = {
        "model": args.model,
        "layer": args.layer,
        "head": args.head,
        "samples_per_label": args.samples_per_label,
        "summary": summary,
        "examples": examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for label in LABELS:
        print(
            label,
            [
                (row["token"], round(row["attention_mass"], 4))
                for row in summary[label]["top_tokens"][:10]
            ],
        )


if __name__ == "__main__":
    main()
