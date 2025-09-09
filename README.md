# effective-llm-ios

Utilities for running optimized language models on iOS devices with key-value cache support and Apple Neural Engine acceleration.

## Features

✅ **Enhanced EXAONE-4.0 Support**: Optimized conversion for LG AI EXAONE models  
✅ **Key-Value Cache**: Persistent attention state for efficient autoregressive generation  
✅ **Apple Neural Engine**: Optimized tensor layouts and operations for maximum ANE utilization  
✅ **Modern Quantization**: Block-wise int4 and float16 quantization with latest coremltools  
✅ **Multi-Architecture**: Support for different transformer attention patterns  
✅ **iOS17+ Optimized**: Updated for latest CoreML and Neural Engine features  

## Python conversion script

`scripts/convert_to_coreml.py` downloads a Hugging Face transformer and applies
advanced Apple Neural Engine optimizations before exporting a CoreML
`mlprogram` with key-value cache support. The enhanced tool:

- **Fused attention**: Uses `torch.nn.functional.scaled_dot_product_attention` with explicit KV cache
- **ANE optimization**: Converts tensors to (B, C, 1, S) format and uses Conv2D operations
- **Advanced quantization**: Block-wise int4 quantization or float16 precision
- **Multi-model support**: Handles different transformer architectures automatically
- **Enhanced error handling**: Robust conversion with detailed logging

### Usage Examples

```bash
# Convert EXAONE-4.0-1.2B with float16 precision
python scripts/convert_to_coreml.py \
    --model-id "LGAI-EXAONE/EXAONE-4.0-1.2B" \
    --max-seq-len 32 \
    --precision float16 \
    --output EXAONE_4_0_1_2B.mlpackage

# Convert with int4 quantization for smaller size
python scripts/convert_to_coreml.py \
    --model-id "LGAI-EXAONE/EXAONE-4.0-1.2B" \
    --precision int4 \
    --output EXAONE_4_0_1_2B_int4.mlpackage \
    --verbose
```

## iOS demo application

`ios/EffectiveLLMApp` is an enhanced SwiftUI application with advanced features:

- **KV Cache Management**: Automatic state preservation between generations
- **Temperature Sampling**: Configurable text generation parameters  
- **Neural Engine Targeting**: Optimized CoreML prediction options
- **Error Handling**: Robust inference with graceful fallbacks
- **EXAONE Tokenizer**: Downloads and uses the official EXAONE tokenizer

The app bridges to Objective-C for CoreML inference and provides an improved
text generation interface with proper attention state management.

## Technical Improvements

### Key-Value Cache Support
- Persistent attention state using CoreML StateType
- Automatic cache initialization and management
- Memory-efficient cache concatenation
- Proper state reset for new conversations

### Apple Neural Engine Optimization
- Conv2D operations instead of linear layers
- Channel-first tensor format (B, C, 1, S)
- Fused scaled dot-product attention
- iOS17+ deployment target for latest features

### Modern Quantization
- Block-wise int4 quantization using coremltools optimize module
- Hardware-accelerated float16 precision
- Fallback to legacy quantization methods
- Quality-preserving compression techniques

See [ENHANCED_FEATURES.md](ENHANCED_FEATURES.md) for detailed technical information.
