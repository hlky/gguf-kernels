# Repository Guidelines

## Project Structure & Module Organization
`gguf_kernels/` is the package root. `gguf_kernels/ops.py` exposes the PyTorch operator wrapper and fake registration for `_C_gguf::ggml_dequantize`. Native sources live in `gguf_kernels/csrc/`: `torch_bindings.cpp` registers the op, `kernel.cu` contains CUDA entry points, and the headers (`common.h`, `dequantize.h`, `dispatch_utils.h`) hold GGUF quantization logic and helpers. Root-level `gguf_np.py` and `gguf_pt.py` provide NumPy and PyTorch-side quantization utilities.

## Build, Test, and Development Commands
Use a Python 3.10+ environment with PyTorch and a working CUDA toolchain.

- `python setup.py build_ext --inplace` builds the `_C_gguf` extension in place for local iteration.
- `python -m compileall gguf_kernels gguf_np.py gguf_pt.py` performs a quick syntax check for Python code.
- `python -c "import gguf_np, gguf_pt, gguf_kernels.ops"` is the minimum smoke test after Python-side changes.

If you edit packaging metadata, validate both `setup.py` and `pyproject.toml` before opening a PR.

## Coding Style & Naming Conventions
Follow the existing style in each layer. Python uses 4-space indentation, snake_case function names, and explicit type hints where practical. Keep quantization names aligned with GGML/GGUF identifiers such as `Q4_0`, `Q6_K`, and `IQ2_XS`. C++/CUDA code should stay consistent with the current header layout, use descriptive helper names, and avoid mixing unrelated kernel changes into one patch.

## Testing Guidelines
There is no committed `tests/` suite yet, so every change should include at least one manual validation path. Prefer adding focused tests under a new `tests/` directory when touching numerical behavior. Name Python tests `test_<feature>.py`. For kernel changes, document:

- quantization types exercised
- tensor shapes and dtypes used
- CPU vs. CUDA comparison or expected tolerance

## Commit & Pull Request Guidelines
The current history uses short imperative subjects (`init`). Keep commit titles concise, imperative, and under 72 characters, for example `Add Q5_K CUDA dequant path`. PRs should explain what changed, which quant types or kernels were affected, the environment used for validation, and any accuracy or performance impact. Include exact commands used for manual testing in the PR body.
