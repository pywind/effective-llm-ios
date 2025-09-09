# Enhanced PyTorch to CoreML Conversion for EXAONE-4.0

This repository contains an optimized conversion pipeline for converting Hugging Face transformer models (specifically EXAONE-4.0) to CoreML with key-value cache support and Apple Neural Engine acceleration.

## Key Features

### ✅ Enhanced Model Support
- **Multi-architecture compatibility**: Supports different attention patterns (standard q/k/v projections, combined qkv, etc.)
- **EXAONE-4.0 optimized**: Specifically tested and optimized for LG AI EXAONE models
- **Robust model loading**: Better error handling and model validation

### ✅ Advanced Apple Neural Engine Optimization
- **Fused SDPA**: Uses `torch.nn.functional.scaled_dot_product_attention` for optimal performance
- **Conv2D conversion**: Linear layers converted to 1x1 Conv2D for better ANE utilization
- **Optimal tensor layout**: (B, C, 1, S) format for minimal memory copies on ANE
- **iOS17 targeting**: Updated deployment target for latest ANE features

### ✅ Key-Value Cache Implementation
- **Persistent KV cache**: Explicit StateType inputs/outputs for maintaining attention state
- **Automatic cache management**: iOS code automatically handles cache initialization and updates
- **Memory efficient**: Cache concatenation along sequence dimension

### ✅ Modern Quantization Support
- **Block-wise int4 quantization**: Using latest coremltools optimize module
- **Float16 support**: Hardware-accelerated 16-bit precision
- **Fallback compatibility**: Graceful degradation to legacy quantization methods

## Usage

### Converting Models

```bash
# Convert EXAONE-4.0-1.2B model with float16 precision
python scripts/convert_to_coreml.py \
    --model-id "LGAI-EXAONE/EXAONE-4.0-1.2B" \
    --max-seq-len 32 \
    --precision float16 \
    --output EXAONE_4_0_1_2B.mlpackage \
    --verbose

# Convert with int4 quantization for smaller model size
python scripts/convert_to_coreml.py \
    --model-id "LGAI-EXAONE/EXAONE-4.0-1.2B" \
    --max-seq-len 32 \
    --precision int4 \
    --output EXAONE_4_0_1_2B_int4.mlpackage
```

### iOS Integration

The iOS app now includes:
- **Automatic KV cache management**: State is preserved between generations
- **Temperature sampling**: Configurable text generation parameters
- **Better error handling**: Graceful fallbacks and detailed logging
- **Neural Engine targeting**: Optimized prediction options

## Technical Improvements

### Attention Mechanism
- **Multi-pattern support**: Handles different transformer architectures
- **Dimension compatibility**: Automatic handling of 2D, 3D, and 4D tensors
- **Causal masking**: Proper attention masking for autoregressive generation

### CoreML Conversion
- **Latest APIs**: Uses newest coremltools features where available
- **Compute unit optimization**: Proper ANE targeting
- **State management**: Explicit KV cache state handling
- **Model validation**: Post-conversion checks and logging

### iOS Performance
- **Cache reuse**: Persistent attention state between predictions
- **Memory efficiency**: Proper cache initialization and management
- **Generation control**: Configurable sampling parameters
- **Error resilience**: Robust error handling throughout the pipeline

## Architecture Details

### Tensor Flow
1. **Input**: Text → Tokens (Int32)
2. **Embedding**: Tokens → Hidden States (B, C, 1, S)
3. **Attention**: Fused SDPA with KV cache
4. **Output**: Logits for next token prediction
5. **Cache**: Updated KV states for next iteration

### ANE Optimization Principles
1. **Channel-first format**: (B, C, H, W) tensor layout
2. **Conv2D operations**: 1x1 convolutions instead of linear layers
3. **Fixed shapes**: Consistent tensor dimensions for optimal compilation
4. **Minimal data movement**: Efficient memory access patterns

## Dependencies

- **transformers**: 4.56.1+ (latest features and EXAONE support)
- **coremltools**: 8.3.0+ (latest optimization modules)
- **torch**: 2.8.0+ (fused attention support)

## Supported Models

- LGAI-EXAONE/EXAONE-4.0-1.2B
- LGAI-EXAONE/EXAONE-4.0-7B (with appropriate memory configuration)
- Other transformer models with standard attention patterns

## Performance Notes

- **Int4 quantization**: ~75% size reduction with minimal quality loss
- **Float16**: ~50% size reduction, better precision
- **KV cache**: Significantly faster autoregressive generation
- **ANE acceleration**: Up to 10x speedup on supported devices