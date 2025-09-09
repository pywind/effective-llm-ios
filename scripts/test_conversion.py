#!/usr/bin/env python3
"""
Test script to validate the conversion functionality.
"""

import torch
import torch.nn as nn
from convert_to_coreml import FusedSDPAWithKVCache
import logging

logging.basicConfig(level=logging.INFO)

class MockConfig:
    def __init__(self):
        self.num_attention_heads = 8
        self.hidden_size = 512
        self.head_dim = 64

class MockAttention(nn.Module):
    def __init__(self, hidden_size=512, num_heads=8):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

def test_fused_attention():
    """Test the FusedSDPAWithKVCache module."""
    print("Testing FusedSDPAWithKVCache...")
    
    config = MockConfig()
    mock_attn = MockAttention()
    
    # Create fused attention module
    fused_attn = FusedSDPAWithKVCache(mock_attn, config)
    
    # Test with different input formats
    batch_size, seq_len, hidden_size = 1, 4, 512
    
    # Test 1: 3D input (B, S, C)
    hidden_states_3d = torch.randn(batch_size, seq_len, hidden_size)
    output, key_cache, value_cache = fused_attn(hidden_states_3d)
    print(f"3D input test passed. Output shape: {output.shape}")
    
    # Test 2: 4D input (B, C, 1, S) - already converted
    hidden_states_4d = hidden_states_3d.transpose(1, 2).unsqueeze(2)
    output, key_cache, value_cache = fused_attn(hidden_states_4d)
    print(f"4D input test passed. Output shape: {output.shape}")
    
    # Test 3: With KV cache
    output2, key_cache2, value_cache2 = fused_attn(hidden_states_4d, key_cache, value_cache)
    print(f"KV cache test passed. Cache shape: {key_cache2.shape}")
    
    print("All tests passed!")

if __name__ == "__main__":
    test_fused_attention()