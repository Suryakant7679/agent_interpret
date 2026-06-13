#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from tool_circuits import LABELS
from tool_circuits.hf_utils import resolve_decoder_layers, resolve_mlp
from tool_circuits.io import read_jsonl
from tool_circuits.prompting import format_tool_prompt
from tool_circuits.sampling import balanced_subset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract final-prompt-token residual and MLP activations"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model weights with bitsandbytes 4-bit quantization.",
    )
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Install research dependencies: python3 -m pip install -e '.[research]'"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model_kwargs = {
        "torch_dtype": "auto",
        "device_map": args.device_map,
    }
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
    mlp_cache: list[torch.Tensor | None] = [None] * len(layers)
    handles = []

    for layer_index, layer in enumerate(layers):
        def capture(_module, _inputs, output, index=layer_index):
            tensor = output[0] if isinstance(output, tuple) else output
            mlp_cache[index] = tensor[:, -1, :].detach().float().cpu()

        handles.append(resolve_mlp(layer).register_forward_hook(capture))

    rows = balanced_subset(list(read_jsonl(args.input)), args.limit)
    selected_counts = {
        label: sum(row["label"] == label for row in rows) for label in LABELS
    }
    print(f"Selected balanced activation subset: {selected_counts}")
    residual_samples = []
    mlp_samples = []
    labels = []
    ids = []

    try:
        for index, row in enumerate(rows, start=1):
            prompt = format_tool_prompt(row["query"])
            messages = [{"role": "user", "content": prompt}]
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
            mlp_cache[:] = [None] * len(layers)
            with torch.inference_mode():
                output = model(**inputs, output_hidden_states=True, use_cache=False)
            residual = torch.stack(
                [state[:, -1, :].detach().float().cpu() for state in output.hidden_states]
            ).squeeze(1)
            if any(item is None for item in mlp_cache):
                raise RuntimeError("One or more MLP hooks did not fire")
            mlp = torch.stack([item for item in mlp_cache if item is not None]).squeeze(1)
            residual_samples.append(residual.numpy())
            mlp_samples.append(mlp.numpy())
            labels.append(LABELS.index(row["label"]))
            ids.append(row["id"])
            print(f"\rExtracted {index}/{len(rows)}", end="", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    print()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        residual=np.stack(residual_samples),
        mlp=np.stack(mlp_samples),
        labels=np.asarray(labels, dtype=np.int64),
        ids=np.asarray(ids),
        label_names=np.asarray(LABELS),
        model=np.asarray(args.model),
        position=np.asarray("final_prompt_token"),
    )
    print(
        f"Saved residual {np.stack(residual_samples).shape} and "
        f"MLP {np.stack(mlp_samples).shape} to {args.output}"
    )


if __name__ == "__main__":
    main()
