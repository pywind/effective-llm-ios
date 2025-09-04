# effective-llm-ios

Utilities for running an optimized language model on iOS devices.

## Python conversion script

`scripts/convert_to_coreml.py` downloads a Hugging Face transformer and applies
several Apple Neural Engine focused optimizations before exporting a CoreML
`mlprogram` with fixed input and output sizes. The tool:

- rewrites attention layers to use fused scaled dot-product attention with an
  explicit key/value cache
- arranges tensors in the (B, C, 1, S) data format for efficient memory access
- optionally quantizes weights with block-wise int4 precision or float16

## iOS demo application

`ios/EffectiveLLMApp` is a minimal SwiftUI application. The app bridges to
Objective-C for the CoreML inference call and provides a simple text in/
text out interface. At launch the app downloads the
`LGAI-EXAONE/EXAONE-4.0-1.2B` tokenizer from Hugging Face to perform
proper tokenization and detokenization.
