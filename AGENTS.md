# Repository Guidelines

## Project Structure & Module Organization

`gguf_kernels/` contains the installable CUDA extension package. `gguf_kernels/ops.py` exposes the PyTorch op wrapper and fake registration for `_C_gguf::ggml_dequantize`; `gguf_kernels/formats.py` holds shared GGML/GGUF enum and block-size metadata; `gguf_kernels/ggml_ref.py` contains small reference helpers. Native sources live in `gguf_kernels/csrc/`, with `torch_bindings.cpp` registering the op, `kernel.cu` dispatching kernels, and headers such as `common.h`, `dequantize.h`, and `dispatch_utils.h` holding format logic.

Root-level `gguf_np.py` and `gguf_pt.py` are reference implementations used by tests and benchmarks. `libgguf/` is a separate Python package with minimized llama.cpp-derived quantizer bindings. `scripts/` contains build and benchmark utilities. `tests/` contains accuracy, metadata, missing-quantizer, and `libgguf` reference tests. Treat `third_party/llama.cpp/` as vendored upstream reference material.

## Build, Test, and Development Commands

Use Python 3.10+ with PyTorch installed for the target CUDA version.

- `python -m pip install -e .` installs the main package in editable mode.
- `python -m pip install -e libgguf --no-build-isolation` installs the reference quantizer package.
- `python setup.py build_ext --inplace` builds the `_C_gguf` CUDA extension for local iteration.
- `python scripts/build_libgguf.py` builds the standalone shared `libgguf` library.
- `python -m pytest` runs the full test suite; CUDA tests skip when CUDA or the extension is unavailable.
- `python scripts/benchmark_dequant_cuda.py --rows 2048 --cols 4096 --dtype float16 --check` runs a representative CUDA benchmark.
- `python -m compileall gguf_kernels gguf_np.py gguf_pt.py libgguf scripts tests` performs a quick syntax check.

## Coding Style & Naming Conventions

Follow the style already present in each layer. Python uses 4-space indentation, snake_case names, and type hints where they make interfaces clearer. Keep quantization identifiers aligned with upstream GGML names such as `Q4_0`, `Q6_K`, `IQ2_XS`, and `NVFP4`. C++/CUDA changes should stay close to the existing header organization, use C++17-compatible code, and avoid mixing unrelated format or dispatch edits into one patch.

Keep format metadata synchronized across `gguf_kernels/formats.py`, `gguf_np.py`, `gguf_pt.py`, CUDA dispatch, and `FORMAT_SUPPORT.md`. Do not silently change block sizes, type sizes, or enum values without adding tests that prove the new metadata.

## Testing Guidelines

Use `gguf_np` and `libgguf` as correctness references before changing CUDA kernels. For numerical changes, add focused tests under `tests/` and include the quantization types, tensor shapes, dtypes, and tolerances being exercised. Prefer updating existing accuracy fixtures in `tests/accuracy_utils.py` rather than duplicating tensor-generation logic.

Run targeted tests first:

- Metadata or shape helpers: `python -m pytest tests/test_formats_metadata.py`
- NumPy reference behavior: `python -m pytest tests/test_gguf_np_accuracy.py`
- Torch fallback/native behavior: `python -m pytest tests/test_gguf_pt_accuracy.py`
- CUDA kernel behavior: `python -m pytest tests/test_gguf_kernels_accuracy.py`
- Reference quantizer bindings: `python -m pytest tests/test_libgguf_reference.py`

For CUDA kernel changes, build with `python setup.py build_ext --inplace` before running tests or benchmarks.

## Commit & Pull Request Guidelines

Keep commit subjects concise, imperative, and under 72 characters, for example `Add Q5_K CUDA dequant path`. Pull requests should explain what changed, which quantization formats or kernels were affected, the validation environment, and exact test or benchmark commands used. Include accuracy or performance impact when relevant.

## Agent-Specific Notes

Avoid broad edits under `third_party/llama.cpp/`; use it for reference unless explicitly asked to update vendored code. Preserve unrelated worktree changes. When touching generated build outputs, prefer rebuilding from source and do not commit cache directories such as `build/`, `__pycache__/`, or `.pytest_cache/`.
