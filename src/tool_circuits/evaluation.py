from __future__ import annotations

import json
import os
import subprocess
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from .prompting import format_tool_prompt


class Backend(Protocol):
    def generate(self, prompt: str) -> str: ...


@dataclass
class OllamaBackend:
    model: str
    host: str = "http://127.0.0.1:11434"
    timeout: int = 120
    use_cli: bool = False

    def generate(self, prompt: str) -> str:
        if self.use_cli:
            env = os.environ.copy()
            process = subprocess.run(
                ["ollama", "run", self.model, prompt],
                check=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
            )
            return process.stdout.strip()

        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0, "num_predict": 8},
            }
        ).encode()
        request = urllib.request.Request(
            f"{self.host.rstrip('/')}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)["response"].strip()


class TransformersBackend:
    def __init__(
        self, model: str, device_map: str = "auto", load_in_4bit: bool = False
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Transformers backend requires the 'research' dependencies"
            ) from exc
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        model_kwargs = {
            "device_map": device_map,
            "torch_dtype": "auto",
        }
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        self.model = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)
        self.model.eval()

    def generate(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(rendered, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=8,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


@dataclass
class OracleBackend:
    """Deterministic backend used only for offline pipeline tests."""

    current_label: str = "none"

    def generate(self, prompt: str) -> str:
        return self.current_label


def tool_prompt(query: str) -> str:
    return format_tool_prompt(query)
