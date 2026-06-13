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
from tool_circuits.prompting import format_tool_prompt
from tool_circuits.statistics import bootstrap_mean_interval


def main() -> None:
    parser = argparse.ArgumentParser(description="Steer tool decisions along a direction")
    parser.add_argument("--model", required=True)
    parser.add_argument("--directions", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tool", default="calculator")
    parser.add_argument("--contrast", default="python")
    parser.add_argument(
        "--direction-tool",
        help="Direction to add; defaults to --tool.",
    )
    parser.add_argument(
        "--direction-scale",
        type=float,
        default=1.0,
        help="Multiply the selected direction, e.g. -1 for a sign control.",
    )
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1, 2, 4, 8])
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install research dependencies first.") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    names = ["web_search", "calculator", "python", "none"]
    direction_tool = args.direction_tool or args.tool
    tool_id = tokenizer.encode(args.tool, add_special_tokens=False)
    contrast_id = tokenizer.encode(args.contrast, add_special_tokens=False)
    if len(tool_id) != 1 or len(contrast_id) != 1:
        raise SystemExit("Steering currently requires single-token labels.")
    tool_id, contrast_id = tool_id[0], contrast_id[0]

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
    directions = np.load(args.directions)
    direction_index = args.layer + 1
    if direction_index >= directions.shape[1]:
        raise SystemExit(
            f"Layer {args.layer} requires residual direction index {direction_index}, "
            f"but the archive has only {directions.shape[1]} positions."
        )
    direction = torch.tensor(
        directions[names.index(direction_tool), direction_index],
        device=model.device,
        dtype=torch.float16,
    ) * args.direction_scale

    rows = [
        row
        for row in read_jsonl(args.input)
        if row["label"] == args.tool
        and row.get("prediction", args.contrast) == args.contrast
    ][: args.limit]
    if not rows:
        raise SystemExit("No matching error rows found.")

    def inputs_for(row):
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": format_tool_prompt(row["query"])}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return tokenizer(rendered, return_tensors="pt").to(model.device)

    def difference(logits):
        final = logits[0, -1]
        return float((final[tool_id] - final[contrast_id]).item())

    results = []
    for row_index, row in enumerate(rows, start=1):
        inputs = inputs_for(row)
        with torch.inference_mode():
            baseline = difference(model(**inputs, use_cache=False).logits)
        alpha_rows = []
        for alpha in args.alphas:
            def steer(_module, _inputs, output):
                tensor = output[0] if isinstance(output, tuple) else output
                changed = tensor.clone()
                changed[:, -1, :] += alpha * direction.to(changed.dtype)
                return (changed, *output[1:]) if isinstance(output, tuple) else changed

            handle = layers[args.layer].register_forward_hook(steer)
            try:
                with torch.inference_mode():
                    steered = difference(model(**inputs, use_cache=False).logits)
            finally:
                handle.remove()
            alpha_rows.append(
                {
                    "alpha": alpha,
                    "logit_difference": steered,
                    "effect": steered - baseline,
                    "flipped": steered > 0,
                }
            )
        results.append(
            {"id": row["id"], "query": row["query"], "baseline": baseline, "alphas": alpha_rows}
        )
        print(f"Completed {row_index}/{len(rows)}")

    summary = []
    for alpha in args.alphas:
        selected = [
            item
            for row in results
            for item in row["alphas"]
            if item["alpha"] == alpha
        ]
        mean, low, high = bootstrap_mean_interval(
            np.asarray([item["effect"] for item in selected])
        )
        summary.append(
            {
                "alpha": alpha,
                "mean_effect": mean,
                "effect_ci95": [low, high],
                "flip_rate": float(np.mean([item["flipped"] for item in selected])),
            }
        )

    payload = {
        "model": args.model,
        "tool": args.tool,
        "contrast": args.contrast,
        "direction_tool": direction_tool,
        "direction_scale": args.direction_scale,
        "layer": args.layer,
        "residual_direction_index": direction_index,
        "samples": len(results),
        "summary": summary,
        "examples": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
