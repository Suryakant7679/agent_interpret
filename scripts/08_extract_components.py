#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from tool_circuits import LABELS
from tool_circuits.hf_utils import (
    resolve_attention_output_projection,
    resolve_decoder_layers,
    resolve_down_projection,
)
from tool_circuits.io import read_jsonl
from tool_circuits.prompting import format_tool_prompt
from tool_circuits.sampling import balanced_subset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract true MLP intermediate neurons and per-head outputs"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install the research dependencies first.") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    kwargs = {"torch_dtype": "auto", "device_map": args.device_map}
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
    hidden_size = model.config.hidden_size
    head_dim = hidden_size // num_heads

    neuron_cache = [None] * len(layers)
    head_cache = [None] * len(layers)
    handles = []
    for index, layer in enumerate(layers):
        def capture_neurons(_module, inputs, layer_index=index):
            neuron_cache[layer_index] = inputs[0][:, -1, :].detach().float().cpu()

        def capture_heads(_module, inputs, layer_index=index):
            value = inputs[0][:, -1, :].detach().float().cpu()
            head_cache[layer_index] = value.reshape(
                value.shape[0], num_heads, head_dim
            )

        handles.append(
            resolve_down_projection(layer).register_forward_pre_hook(capture_neurons)
        )
        handles.append(
            resolve_attention_output_projection(layer).register_forward_pre_hook(
                capture_heads
            )
        )

    rows = balanced_subset(list(read_jsonl(args.input)), args.limit)
    neurons, heads, labels, ids = [], [], [], []
    try:
        for row_index, row in enumerate(rows, start=1):
            prompt = format_tool_prompt(row["query"])
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
            neuron_cache[:] = [None] * len(layers)
            head_cache[:] = [None] * len(layers)
            with torch.inference_mode():
                model(**inputs, use_cache=False)
            if any(item is None for item in neuron_cache + head_cache):
                raise RuntimeError("One or more component hooks did not fire")
            neurons.append(
                torch.stack([item for item in neuron_cache if item is not None])
                .squeeze(1)
                .numpy()
            )
            heads.append(
                torch.stack([item for item in head_cache if item is not None])
                .squeeze(1)
                .numpy()
            )
            labels.append(LABELS.index(row["label"]))
            ids.append(row["id"])
            print(f"\rExtracted components {row_index}/{len(rows)}", end="", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    print()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        neurons=np.stack(neurons).astype(np.float16),
        heads=np.stack(heads).astype(np.float16),
        labels=np.asarray(labels, dtype=np.int64),
        ids=np.asarray(ids),
        label_names=np.asarray(LABELS),
        model=np.asarray(args.model),
        num_heads=np.asarray(num_heads),
        head_dim=np.asarray(head_dim),
    )
    print(
        f"Saved neurons {np.stack(neurons).shape} and heads "
        f"{np.stack(heads).shape} to {args.output}"
    )


if __name__ == "__main__":
    main()
