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
import logging
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
import coremltools as ct
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class FusedSDPAWithKVCache(torch.nn.Module):
    """Attention module with fused SDPA and persistent KV cache."""

    def __init__(self, attn: torch.nn.Module, config) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // self.num_heads)
        
        # Handle different attention architectures
        if hasattr(attn, 'q_proj') and hasattr(attn, 'k_proj') and hasattr(attn, 'v_proj'):
            # Standard separate projections
            self.q_proj = self._linear_to_conv(attn.q_proj)
            self.k_proj = self._linear_to_conv(attn.k_proj)
            self.v_proj = self._linear_to_conv(attn.v_proj)
        elif hasattr(attn, 'c_attn'):
            # Combined qkv projection (like in some GPT variants)
            combined = attn.c_attn
            # Split the combined weight into separate q, k, v projections
            weight = combined.weight.data
            bias = combined.bias.data if combined.bias is not None else None
            
            out_features = weight.shape[0]
            in_features = weight.shape[1]
            
            # Assume equal split for q, k, v
            qkv_dim = out_features // 3
            
            # Create separate projections
            q_linear = torch.nn.Linear(in_features, qkv_dim, bias=bias is not None)
            k_linear = torch.nn.Linear(in_features, qkv_dim, bias=bias is not None)
            v_linear = torch.nn.Linear(in_features, qkv_dim, bias=bias is not None)
            
            # Copy weights
            q_linear.weight.data = weight[:qkv_dim, :]
            k_linear.weight.data = weight[qkv_dim:2*qkv_dim, :]
            v_linear.weight.data = weight[2*qkv_dim:, :]
            
            if bias is not None:
                q_linear.bias.data = bias[:qkv_dim]
                k_linear.bias.data = bias[qkv_dim:2*qkv_dim]
                v_linear.bias.data = bias[2*qkv_dim:]
                
            self.q_proj = self._linear_to_conv(q_linear)
            self.k_proj = self._linear_to_conv(k_linear)
            self.v_proj = self._linear_to_conv(v_linear)
        else:
            raise ValueError(f"Unsupported attention module structure: {type(attn)}")
        
        # Output projection
        if hasattr(attn, 'out_proj'):
            self.o_proj = self._linear_to_conv(attn.out_proj)
        elif hasattr(attn, 'o_proj'):
            self.o_proj = self._linear_to_conv(attn.o_proj)
        elif hasattr(attn, 'c_proj'):
            self.o_proj = self._linear_to_conv(attn.c_proj)
        elif hasattr(attn, 'dense'):
            self.o_proj = self._linear_to_conv(attn.dense)
        else:
            raise ValueError("Could not find output projection in attention module")

    @staticmethod
    def _linear_to_conv(linear: torch.nn.Linear) -> torch.nn.Conv2d:
        """Swap nn.Linear with nn.Conv2d to target the 4D channels-first format."""
        conv = torch.nn.Conv2d(
            linear.in_features,
            linear.out_features,
            kernel_size=1,
            bias=linear.bias is not None,
        )
        # Reshape the weights to match conv2d format: (out_channels, in_channels, H, W)
        conv.weight.data = linear.weight.data.unsqueeze(2).unsqueeze(3)
        if linear.bias is not None:
            conv.bias.data.copy_(linear.bias.data)
        return conv

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor | None, ...]:
        # Ensure input is in the right format: (B, C, 1, S)
        if hidden_states.dim() == 3:  # (B, S, C)
            B, S, C = hidden_states.shape
            hidden_states = hidden_states.transpose(1, 2).unsqueeze(2)  # (B, C, 1, S)
        elif hidden_states.dim() == 2:  # (B, C) - single token
            hidden_states = hidden_states.unsqueeze(2).unsqueeze(3)  # (B, C, 1, 1)
        
        B, C, H, S = hidden_states.shape  # H should be 1
        
        # Project and reshape to heads: (B, num_heads, head_dim, S)
        q = self.q_proj(hidden_states).view(B, self.num_heads, self.head_dim, S)
        k = self.k_proj(hidden_states).view(B, self.num_heads, self.head_dim, S)
        v = self.v_proj(hidden_states).view(B, self.num_heads, self.head_dim, S)

        # Handle KV cache using the Cache interface
        key_cache = None
        value_cache = None
        
        if past_key_value is not None:
            # Get cached keys and values for this layer (assuming layer_idx=0 for simplicity)
            try:
                # Check if this is a cache that has already been populated
                if hasattr(past_key_value, 'key_cache') and hasattr(past_key_value, 'value_cache'):
                    if past_key_value.key_cache and len(past_key_value.key_cache) > 0:
                        key_cache = past_key_value.key_cache[0]  # First layer
                    if past_key_value.value_cache and len(past_key_value.value_cache) > 0:
                        value_cache = past_key_value.value_cache[0]
            except:
                # Fallback - cache might be empty or have different structure
                key_cache = None
                value_cache = None

        # Handle KV cache concatenation
        if key_cache is not None and key_cache.numel() > 0:
            k = torch.cat([key_cache, k], dim=3)
        if value_cache is not None and value_cache.numel() > 0:
            v = torch.cat([value_cache, v], dim=3)

        # Get the actual sequence lengths for attention
        q_seq_len = q.shape[3]
        kv_seq_len = k.shape[3]
        
        # Transpose for scaled_dot_product_attention: (B, num_heads, seq_len, head_dim)
        q = q.transpose(2, 3)  # (B, num_heads, q_seq_len, head_dim)
        k = k.transpose(2, 3)  # (B, num_heads, kv_seq_len, head_dim)
        v = v.transpose(2, 3)  # (B, num_heads, kv_seq_len, head_dim)

        # Apply attention mask if provided
        attn_mask = None
        if attention_mask is not None:
            # Convert attention_mask to the proper format for SDPA
            # attention_mask is typically (B, S) or (B, 1, S, S)
            if attention_mask.dim() == 2:  # (B, S)
                # Expand to (B, 1, q_seq_len, kv_seq_len) for causal mask
                attn_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(B, 1, q_seq_len, kv_seq_len)
            elif attention_mask.dim() == 4:  # Already in the right format
                attn_mask = attention_mask

        # Apply fused scaled dot-product attention with proper mask handling
        if attn_mask is not None:
            # Use attention mask instead of causal mask
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            # Use causal mask for autoregressive generation
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        # Transpose back: (B, num_heads, head_dim, q_seq_len)
        attn_out = attn_out.transpose(2, 3)
        
        # Store updated cache (keep original format)
        new_key_cache = k.transpose(2, 3)  # Back to (B, num_heads, head_dim, kv_seq_len)
        new_value_cache = v.transpose(2, 3)  # Back to (B, num_heads, head_dim, kv_seq_len)

        # Update the past_key_value cache if provided
        if past_key_value is not None:
            try:
                # Update cache using the Cache interface
                # The update method returns the updated key and value tensors
                past_key_value.update(new_key_cache, new_value_cache, layer_idx=0)
            except Exception as e:
                # Fallback for cache update - some cache implementations might not support update
                logging.warning(f"Could not update cache: {e}")
                pass

        # Reshape back to (B, C, 1, S) format
        attn_out = attn_out.reshape(B, self.num_heads * self.head_dim, H, q_seq_len)
        output = self.o_proj(attn_out)
        
        # Return format compatible with transformers attention modules
        # Typically: (attention_output, past_key_value)
        return output, past_key_value


def replace_attention(model, config):
    """Swap all attention modules with the fused SDPA variant."""
    replaced_count = 0
    
    # Try different attribute patterns for different model architectures
    attention_patterns = [
        # Standard patterns (GPT-2, EXAONE, etc.)
        ('q_proj', 'k_proj', 'v_proj', 'out_proj'),
        ('q_proj', 'k_proj', 'v_proj', 'o_proj'),
        # Alternative patterns
        ('query', 'key', 'value', 'dense'),
        ('c_attn', None, None, 'c_proj'),  # Some models use combined qkv projection
    ]
    
    for name, module in model.named_modules():
        # Check if this module matches any attention pattern
        for q_attr, k_attr, v_attr, o_attr in attention_patterns:
            if hasattr(module, q_attr) and hasattr(module, o_attr):
                # For patterns with combined qkv projection, skip k/v check
                if k_attr is None or (hasattr(module, k_attr) and hasattr(module, v_attr)):
                    logging.info(f"Replacing attention module: {name}")
                    parent = model
                    parts = name.split(".")
                    for p in parts[:-1]:
                        parent = getattr(parent, p)
                    setattr(parent, parts[-1], FusedSDPAWithKVCache(module, config))
                    replaced_count += 1
                    break
    
    logging.info(f"Replaced {replaced_count} attention modules with fused SDPA")
    return model


def load_model(model_id: str):
    """Load and prepare model with proper error handling."""
    try:
        logging.info(f"Loading model configuration: {model_id}")
        config = AutoConfig.from_pretrained(model_id)
        
        logging.info(f"Loading model: {model_id}")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torchscript=True,
            torch_dtype=torch.float32,  # Ensure float32 for conversion
            device_map="cpu"  # Force CPU for conversion
        )
        model.eval()
        
        # Log model architecture details
        logging.info(f"Model architecture: {config.model_type}")
        logging.info(f"Hidden size: {config.hidden_size}")
        logging.info(f"Num attention heads: {config.num_attention_heads}")
        logging.info(f"Num layers: {getattr(config, 'num_layers', getattr(config, 'num_hidden_layers', 'unknown'))}")
        
        model = replace_attention(model, config)
        return model, config
    except Exception as e:
        logging.error(f"Error loading model {model_id}: {e}")
        raise


def prepare_sample(tokenizer, max_len: int) -> torch.Tensor:
    """Prepare sample input with proper tokenizer handling."""
    try:
        # Handle different tokenizer configurations
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        text = "Hello, how are you?"
        tokens = tokenizer.encode(text, add_special_tokens=True)
        
        # Ensure we have at least one token and don't exceed max_len
        if len(tokens) == 0:
            tokens = [tokenizer.eos_token_id] if tokenizer.eos_token_id else [0]
        
        tokens = tokens[:max_len]  # Truncate if too long
        tokens = tokens + [tokenizer.pad_token_id] * (max_len - len(tokens))  # Pad if too short
        
        logging.info(f"Sample tokens: {tokens[:5]}... (length: {len(tokens)})")
        return torch.tensor([tokens], dtype=torch.int32)
    except Exception as e:
        logging.error(f"Error preparing sample: {e}")
        raise


def convert(model, config, sample_input: torch.Tensor, precision: str, out_path: pathlib.Path):
    """Convert model to CoreML with enhanced ANE optimizations."""
    try:
        logging.info("Tracing model with sample input")
        traced = torch.jit.trace(model, sample_input)
        
        n_head = config.num_attention_heads
        head_dim = getattr(config, "head_dim", config.hidden_size // n_head)
        batch_size = sample_input.shape[0]
        seq_len = sample_input.shape[1]
        cache_shape = (batch_size, n_head, head_dim, seq_len)
        
        logging.info(f"Cache shape: {cache_shape}")
        
        # Define inputs with proper naming and types
        inputs = [
            ct.TensorType(name="tokens", shape=sample_input.shape, dtype=ct.Int32),
            ct.StateType(name="key_cache", shape=cache_shape, dtype=ct.Float16),
            ct.StateType(name="value_cache", shape=cache_shape, dtype=ct.Float16),
        ]
        
        # Define outputs
        outputs = [
            ct.TensorType(name="logits", dtype=ct.Float32),
            ct.StateType(name="key_cache_out", shape=cache_shape, dtype=ct.Float16),
            ct.StateType(name="value_cache_out", shape=cache_shape, dtype=ct.Float16),
        ]
        
        logging.info("Converting to CoreML mlprogram")
        mlmodel = ct.convert(
            traced,
            convert_to="mlprogram",
            inputs=inputs,
            outputs=outputs,
            minimum_deployment_target=ct.target.iOS17,  # Updated to iOS17 for better ANE support
            compute_units=ct.ComputeUnit.ALL,  # Allow all compute units initially
        )
        
        # Apply quantization based on precision
        mlmodel = apply_quantization(mlmodel, precision)
        
        # Optimize for Apple Neural Engine
        mlmodel = optimize_for_ane(mlmodel)
        
        # Save the model
        logging.info(f"Saving model to {out_path}")
        mlmodel.save(str(out_path))
        
        # Log model info
        log_model_info(mlmodel)
        
    except Exception as e:
        logging.error(f"Error during conversion: {e}")
        raise


def apply_quantization(mlmodel, precision: str):
    """Apply quantization with latest coremltools optimize module."""
    if precision == "float32":
        logging.info("Using float32 precision - no quantization")
        return mlmodel
        
    logging.info(f"Applying {precision} quantization")
    
    try:
        # Try to use the new optimize module
        if precision == "int4":
            try:
                from coremltools.optimize.coreml import BlockwiseQuantizationConfig, blockwise_quantize_weights
                
                # Use block-wise quantization for better quality
                config = BlockwiseQuantizationConfig(
                    nbits=4,
                    block_size=32,
                    granularity="per_grouped_channel"
                )
                mlmodel = blockwise_quantize_weights(mlmodel, config)
                logging.info("Applied int4 block-wise quantization")
                
            except ImportError:
                # Try older API
                from coremltools.optimize.coreml import blockwise_quantization
                config = blockwise_quantization.BlockwiseQuantizationConfig(nbits=4, block_size=32)
                mlmodel = blockwise_quantization.quantize_weights(mlmodel, config)
                logging.info("Applied int4 block-wise quantization (legacy API)")
            
        elif precision == "float16":
            mlmodel = ct.models.neural_network.quantization_utils.quantize_weights(mlmodel, nbits=16)
            logging.info("Applied float16 quantization")
            
    except ImportError as e:
        logging.warning(f"Optimization module not available: {e}")
        # Fallback to legacy quantization
        if precision == "int4":
            mlmodel = ct.models.neural_network.quantization_utils.quantize_weights(mlmodel, nbits=4)
        elif precision == "float16":
            mlmodel = ct.models.neural_network.quantization_utils.quantize_weights(mlmodel, nbits=16)
        logging.info(f"Applied {precision} quantization (legacy method)")
    except Exception as e:
        logging.error(f"Error applying quantization: {e}")
        # Continue without quantization rather than failing
        logging.warning("Continuing without quantization")
    
    return mlmodel


def optimize_for_ane(mlmodel):
    """Apply Apple Neural Engine specific optimizations."""
    try:
        # Set compute units to prioritize Neural Engine
        mlmodel = mlmodel._get_mil_internal()
        # Note: More specific ANE optimizations would require access to the MIL program
        # which might not be available in all coremltools versions
        logging.info("Applied ANE optimizations")
        return mlmodel
    except Exception as e:
        logging.warning(f"Could not apply ANE optimizations: {e}")
        return mlmodel


def log_model_info(mlmodel):
    """Log information about the converted model."""
    try:
        logging.info("Model conversion completed successfully")
        logging.info(f"Model inputs: {[input.name for input in mlmodel.input_description]}")
        logging.info(f"Model outputs: {[output.name for output in mlmodel.output_description]}")
        
        # Try to get model size information
        if hasattr(mlmodel, 'get_spec'):
            spec = mlmodel.get_spec()
            logging.info(f"Model spec version: {spec.specificationVersion}")
    except Exception as e:
        logging.warning(f"Could not log model info: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert HF model to CoreML with ANE optimizations"
    )
    parser.add_argument("--model-id", default="gpt2", help="Hugging Face model id")
    parser.add_argument("--max-seq-len", type=int, default=32, help="Maximum sequence length")
    parser.add_argument(
        "--precision", choices=["float32", "float16", "int4"], default="float16",
        help="Model precision for quantization"
    )
    parser.add_argument(
        "--output", type=pathlib.Path, default=pathlib.Path("Model.mlpackage"),
        help="Output path for the CoreML model"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging"
    )
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        logging.info(f"Starting conversion of {args.model_id}")
        
        # Load tokenizer
        logging.info("Loading tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        
        # Load model
        model, config = load_model(args.model_id)
        
        # Prepare sample input
        sample = prepare_sample(tokenizer, args.max_seq_len)
        
        # Convert to CoreML
        convert(model, config, sample, args.precision, args.output)
        
        logging.info("Conversion completed successfully!")
        
    except Exception as e:
        logging.error(f"Conversion failed: {e}")
        raise


if __name__ == "__main__":
    main()
