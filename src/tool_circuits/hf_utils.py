from __future__ import annotations

from collections.abc import Sequence


def resolve_decoder_layers(model) -> Sequence:
    candidates = [
        ("model", "layers"),
        ("transformer", "h"),
        ("model", "decoder", "layers"),
    ]
    for path in candidates:
        current = model
        try:
            for attribute in path:
                current = getattr(current, attribute)
        except AttributeError:
            continue
        return current
    raise ValueError(
        "Unsupported architecture: could not locate decoder layers. "
        "Add its module path to resolve_decoder_layers()."
    )


def resolve_mlp(layer):
    for name in ("mlp", "feed_forward", "ffn"):
        if hasattr(layer, name):
            return getattr(layer, name)
    raise ValueError(f"Could not locate MLP module on {type(layer).__name__}")


def resolve_attention(layer):
    for name in ("self_attn", "attention", "attn"):
        if hasattr(layer, name):
            return getattr(layer, name)
    raise ValueError(f"Could not locate attention module on {type(layer).__name__}")


def resolve_down_projection(layer):
    mlp = resolve_mlp(layer)
    for name in ("down_proj", "c_proj", "dense_4h_to_h"):
        if hasattr(mlp, name):
            return getattr(mlp, name)
    raise ValueError(f"Could not locate MLP down projection on {type(mlp).__name__}")


def resolve_attention_output_projection(layer):
    attention = resolve_attention(layer)
    for name in ("o_proj", "out_proj", "c_proj"):
        if hasattr(attention, name):
            return getattr(attention, name)
    raise ValueError(
        f"Could not locate attention output projection on {type(attention).__name__}"
    )
