# Format Support Matrix

This matrix is derived from the current implementation in `gguf_kernels/csrc/dequantize.h`, `gguf_pt.py`, and `gguf_np.py`.

## Summary

- `gguf_kernels` provides native CUDA dequantization for 19 quantized GGUF formats via `torch.ops._C_gguf.ggml_dequantize`.
- `gguf_pt` provides native Torch implementations for 14 quantized formats. `dequantize_tensor()` also treats `F32` and `F16` as pass-through and falls back to `gguf_np.dequantize()` for formats that are not native Torch.
- `gguf_np` is the widest reference implementation here and currently dequantizes 27 formats.
- `gguf_np.quantize()` now covers the previously missing K-quants and IQ formats by delegating those encoders to the separate `libgguf` Python package. Install it with `python -m pip install -e libgguf --no-build-isolation`.

## Native Support by Format

| Format | gguf_kernels | gguf_pt | gguf_np | Notes |
| --- | --- | --- | --- | --- |
| `F32` | no | pass-through | yes | Dense float data, not a CUDA kernel format here |
| `F16` | no | pass-through | yes | Dense half data, not a CUDA kernel format here |
| `Q1_0` | no | yes | yes | Upstream GGML format now covered in Python layers |
| `Q4_0` | yes | yes | yes |  |
| `Q4_1` | yes | yes | yes |  |
| `Q5_0` | yes | yes | yes |  |
| `Q5_1` | yes | yes | yes |  |
| `Q8_0` | yes | yes | yes |  |
| `Q8_1` | no | no | no | Present in enum and size table only |
| `Q2_K` | yes | yes | yes |  |
| `Q3_K` | yes | yes | yes |  |
| `Q4_K` | yes | yes | yes |  |
| `Q5_K` | yes | yes | yes |  |
| `Q6_K` | yes | yes | yes |  |
| `Q8_K` | no | no | no | Present in enum and size table only |
| `IQ2_XXS` | yes | no | yes | `gguf_pt.dequantize_tensor()` falls back to `gguf_np` |
| `IQ2_XS` | yes | no | yes | `gguf_pt.dequantize_tensor()` falls back to `gguf_np` |
| `IQ3_XXS` | yes | no | yes | `gguf_pt.dequantize_tensor()` falls back to `gguf_np` |
| `IQ1_S` | yes | no | yes | `gguf_pt.dequantize_tensor()` falls back to `gguf_np` |
| `IQ4_NL` | yes | yes | yes |  |
| `IQ3_S` | yes | no | yes | `gguf_pt.dequantize_tensor()` falls back to `gguf_np` |
| `IQ2_S` | yes | no | yes | `gguf_pt.dequantize_tensor()` falls back to `gguf_np` |
| `IQ4_XS` | yes | yes | yes |  |
| `I8` | no | no | no | Present in enum and size table only |
| `I16` | no | no | no | Present in enum and size table only |
| `I32` | no | no | no | Present in enum and size table only |
| `I64` | no | no | no | Present in enum and size table only |
| `F64` | no | no | no | Present in enum and size table only |
| `IQ1_M` | yes | no | yes | `gguf_pt.dequantize_tensor()` falls back to `gguf_np` |
| `BF16` | no | yes | yes | Torch and NumPy handle this without CUDA kernels here |
| `TQ1_0` | no | no | yes | Reference-only in this repo |
| `TQ2_0` | no | no | yes | Reference-only in this repo |
| `MXFP4` | no | no | yes | Reference-only in this repo |
| `NVFP4` | no | yes | yes | Upstream GGML format now covered in Python layers |

## Notes

- The matrix above tracks native implementations, not enum coverage. `GGMLQuantizationType` and `GGML_QUANT_SIZES` include types that do not yet have working dequantization code in this repo.
- For `gguf_np`, dequantization remains self-contained NumPy. Quantization for `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ1_S`, `IQ1_M`, `IQ4_NL`, and `IQ4_XS` now uses the separate `libgguf` package.
- For `gguf_pt`, use `dequantize_tensor()` if you want automatic fallback to `gguf_np` for non-native formats. The lower-level `dequantize()` helper only works for entries in `dequantize_functions`.
