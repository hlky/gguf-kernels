# gguf-kernels

Exploration and benchmarking of GGUF dequantization kernels for CUDA, plus Python reference implementations used to check correctness. The end goal is to bring GGUF kernel support into [DinoML](https://github.com/hlky/dinoml).

This repository currently contains:

- `gguf_kernels/`: a PyTorch CUDA extension exposing `torch.ops._C_gguf.ggml_dequantize`.
- `gguf_np.py`: NumPy quantize/dequantize reference code with broad GGUF format coverage.
- `gguf_pt.py`: Torch-side dequantization helpers for selected formats, with fallback paths through `gguf_np`.
- `libgguf/`: a separate Python package with CPython bindings for minimized llama.cpp-derived reference quantizers.
- `tests/`: accuracy, metadata, missing-quantizer, and `libgguf` reference checks.
- `scripts/benchmark_dequant_cuda.py`: CUDA dequantization benchmark runner.

See [FORMAT_SUPPORT.md](FORMAT_SUPPORT.md) for the current format support matrix.

## Requirements

- Python 3.10+
- PyTorch
- CUDA toolkit and a CUDA-capable GPU for building and testing `gguf_kernels`
- A C++17 compiler for the `libgguf` reference package

The main package has no runtime dependencies declared in `pyproject.toml`; install PyTorch for your CUDA version before building the extension.

## Setup

Install the local packages in editable mode:

```powershell
python -m pip install -e .
python -m pip install -e libgguf --no-build-isolation
```

Build the CUDA extension in place:

```powershell
python setup.py build_ext --inplace
```

If you only need the standalone shared `libgguf` C ABI artifact used by tests or external callers, build it with:

```powershell
python scripts/build_libgguf.py
```

## Usage

The CUDA operator expects quantized bytes, the GGML quantization type value, and the flattened output row/column shape:

```python
import torch
import gguf_np
from gguf_kernels.ops import ggml_dequantize

qtype = gguf_np.GGMLQuantizationType.Q4_0
reference = torch.randn(2, 128, dtype=torch.float16).numpy()
quantized = gguf_np.quantize(reference, qtype)

out = ggml_dequantize(
    torch.from_numpy(quantized).cuda(),
    qtype.value,
    2,
    128,
    torch.float16,
)
```

Shape helpers live in `gguf_kernels.formats`:

```python
from gguf_kernels.formats import GGMLQuantizationType, quant_shape_to_byte_shape

byte_shape = quant_shape_to_byte_shape((2, 128), GGMLQuantizationType.Q4_0)
```

## Testing

Run the CPU/reference-oriented tests:

```powershell
python -m pytest tests/test_formats_metadata.py tests/test_gguf_np_accuracy.py tests/test_gguf_pt_accuracy.py tests/test_libgguf_reference.py
```

Run the CUDA extension accuracy tests after building the extension:

```powershell
python -m pytest tests/test_gguf_kernels_accuracy.py
```

Run the full suite:

```powershell
python -m pytest
```

Some tests skip automatically when CUDA, the compiled CUDA extension, or the editable `libgguf` package is unavailable.

## Benchmarking

Build the CUDA extension first, then run:

```powershell
python scripts/benchmark_dequant_cuda.py --rows 2048 --cols 4096 --dtype float16 --iters 100 --warmup 25 --check
```

You can benchmark multiple shapes or a subset of quantization formats:

```powershell
python scripts/benchmark_dequant_cuda.py --shapes "512x4096;16x64x8192" --qtypes Q4_0,Q6_K,IQ4_XS
```

The output includes timing, throughput, and effective bandwidth for each shape/format pair.

## Development Notes

- Keep GGUF format metadata synchronized across `gguf_kernels/formats.py`, `gguf_np.py`, `gguf_pt.py`, and the CUDA dispatch code.
- Use `gguf_np` and `libgguf` as correctness references before changing CUDA kernels.
- Future imatrix work could collect llama.cpp-compatible importance vectors from Hugging Face Transformers by registering forward hooks on target linear layers, accumulating column-wise squared input activations over calibration data, and mapping HF module names/layouts back to GGUF tensor names.
- Do not edit vendored `third_party/llama.cpp` files unless the change is explicitly about refreshing or comparing upstream reference behavior.
- When adding a format, update `FORMAT_SUPPORT.md` and add focused tests for shape metadata, quantization/dequantization accuracy, and unsupported-path behavior where relevant.
