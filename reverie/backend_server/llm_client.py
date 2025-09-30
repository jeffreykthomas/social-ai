"""Simple Hugging Face text-generation client for local reasoning models."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


@dataclass
class LLMConfig:
    """Runtime configuration for the local LLM client."""

    model_name: str
    device: Optional[str] = None
    dtype: Optional[str] = "bfloat16"
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.05


class LocalLLMClient:
    """Thread-safe wrapper around a transformers text-generation pipeline."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._lock = threading.Lock()

        torch_dtype = None
        if config.dtype:
            torch_dtype = getattr(torch, config.dtype, None)

        device_map: Any = "auto"
        if config.device:
            device_map = config.device

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.pipeline = pipeline(
            "text-generation",
            model=AutoModelForCausalLM.from_pretrained(
                config.model_name,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            ),
            tokenizer=self.tokenizer,
        )

    def generate(self, messages: List[Dict[str, str]], *,
                 max_new_tokens: Optional[int] = None,
                 temperature: Optional[float] = None,
                 top_p: Optional[float] = None,
                 repetition_penalty: Optional[float] = None) -> str:
        """Generate a chat response given a list of role-content messages."""
        if not messages:
            raise ValueError("messages must contain at least one item")

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        gen_kwargs = {
            "max_new_tokens": max_new_tokens or self.config.max_new_tokens,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "top_p": top_p if top_p is not None else self.config.top_p,
            "repetition_penalty": repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        with self._lock:
            outputs = self.pipeline(prompt, **gen_kwargs)

        generated_text = outputs[0]["generated_text"]
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):]

        return generated_text.strip()


_cached_client: Optional[LocalLLMClient] = None


def get_client(config: LLMConfig) -> LocalLLMClient:
    """Return a cached singleton client keyed by config."""
    global _cached_client
    if _cached_client is None or _cached_client.config != config:
        _cached_client = LocalLLMClient(config)
    return _cached_client
