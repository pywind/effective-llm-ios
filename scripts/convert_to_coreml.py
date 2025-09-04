#!/usr/bin/env python3
"""
Convert a Hugging Face transformer to an ANE-friendly CoreML `mlprogram`.

Key optimizations implemented:
* **Fused scaled dot-product attention** using `torch.nn.functional.scaled_dot_product_attention`.
* **Explicit key/value cache states** with fixed input shapes.
* **Block-wise int4 weight quantization** (fallback to float16).
* Uses the (B, C, 1, S) data format and per-head chunking to minimize memory
  copies on the Apple Neural Engine.
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Tuple

import torch
import torch.nn.functional as F
import coremltools as ct
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


class FusedSDPAWithKVCache(torch.nn.Module):
    """Attention module with fused SDPA and persistent KV cache."""

    def __init__(self, attn: torch.nn.Module, config) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // self.num_heads)
        # Convert linear layers to conv2d to obtain (B, C, 1, S) layout
        self.q_proj = self._linear_to_conv(attn.q_proj)
        self.k_proj = self._linear_to_conv(attn.k_proj)
        self.v_proj = self._linear_to_conv(attn.v_proj)
        self.o_proj = self._linear_to_conv(attn.out_proj)

    @staticmethod
    def _linear_to_conv(linear: torch.nn.Linear) -> torch.nn.Conv2d:
        """Swap nn.Linear with nn.Conv2d to target the 4D channels-first format."""
        conv = torch.nn.Conv2d(
            linear.in_features,
            linear.out_features,
            kernel_size=1,
            bias=linear.bias is not None,
        )
        # Unsqueeze the weights twice to match expected conv2d shape
        conv.weight.data.copy_(linear.weight.data.view(conv.weight.shape))
        if linear.bias is not None:
            conv.bias.data.copy_(linear.bias.data)
        return conv

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_cache: torch.Tensor | None = None,
        value_cache: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # hidden_states expected shape: (B, C, 1, S)
        B, _, _, S = hidden_states.shape
        # Project and chunk into heads (Principle 2)
        q = self.q_proj(hidden_states).view(B, self.num_heads, self.head_dim, S)
        k = self.k_proj(hidden_states).view(B, self.num_heads, self.head_dim, S)
        v = self.v_proj(hidden_states).view(B, self.num_heads, self.head_dim, S)

        if key_cache is not None:
            # Append to caches along sequence axis (Principle 3)
            k = torch.cat([key_cache, k], dim=3)
            v = torch.cat([value_cache, v], dim=3)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        new_key_cache, new_value_cache = k, v

        attn_out = attn_out.reshape(B, self.num_heads * self.head_dim, 1, S)
        output = self.o_proj(attn_out)
        return output, new_key_cache, new_value_cache


def replace_attention(model, config):
    """Swap all attention modules with the fused SDPA variant."""
    for name, module in model.named_modules():
        if hasattr(module, "q_proj") and hasattr(module, "k_proj") and hasattr(module, "v_proj"):
            parent = model
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], FusedSDPAWithKVCache(module, config))
    return model


def load_model(model_id: str):
    config = AutoConfig.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torchscript=True)
    model.eval()
    model = replace_attention(model, config)
    return model, config


def prepare_sample(tokenizer, max_len: int) -> torch.Tensor:
    tokens = tokenizer.encode("Hello", add_special_tokens=False)
    tokens = tokens + [tokenizer.pad_token_id] * (max_len - len(tokens))
    return torch.tensor([tokens], dtype=torch.int32)


def convert(model, config, sample_input: torch.Tensor, precision: str, out_path: pathlib.Path):
    traced = torch.jit.trace(model, sample_input)

    n_head = config.num_attention_heads
    head_dim = getattr(config, "head_dim", config.hidden_size // n_head)
    cache_shape = (1, n_head, head_dim, sample_input.shape[1])

    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="tokens", shape=sample_input.shape, dtype=ct.Int32),
            ct.StateType(name="key_cache", shape=cache_shape, dtype=ct.Float16),
            ct.StateType(name="value_cache", shape=cache_shape, dtype=ct.Float16),
        ],
        outputs=[
            ct.TensorType(name="logits", dtype=ct.Float32),
            ct.StateType(name="key_cache_out", shape=cache_shape, dtype=ct.Float16),
            ct.StateType(name="value_cache_out", shape=cache_shape, dtype=ct.Float16),
        ],
        minimum_deployment_target=ct.target.iOS16,
    )

    if precision == "int4":
        try:
            from coremltools.optimize.coreml import blockwise_quantization

            qconfig = blockwise_quantization.BlockwiseQuantizationConfig(nbits=4, block_size=32)
            mlmodel = blockwise_quantization.quantize_weights(mlmodel, qconfig)
        except Exception:
            # Fallback to legacy quantizer if optimize module is unavailable
            mlmodel = ct.models.neural_network.quantization_utils.quantize_weights(mlmodel, nbits=4)
    elif precision == "float16":
        mlmodel = ct.models.neural_network.quantization_utils.quantize_weights(mlmodel, nbits=16)

    mlmodel.save(str(out_path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert HF model to CoreML with ANE optimizations"
    )
    parser.add_argument("--model-id", default="gpt2", help="Hugging Face model id")
    parser.add_argument("--max-seq-len", type=int, default=32)
    parser.add_argument(
        "--precision", choices=["float32", "float16", "int4"], default="float16"
    )
    parser.add_argument(
        "--output", type=pathlib.Path, default=pathlib.Path("Model.mlpackage")
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model, config = load_model(args.model_id)

    sample = prepare_sample(tokenizer, args.max_seq_len)
    convert(model, config, sample, args.precision, args.output)


if __name__ == "__main__":
    main()
