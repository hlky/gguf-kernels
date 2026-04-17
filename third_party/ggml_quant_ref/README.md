Vendored subset of upstream `ggml` used to build the local GGML quantization reference shared library.

Source:
- Repository: `https://github.com/ggml-org/ggml`
- Commit: `d4fcfe88a8bcf5c9840be14be6c2fbf1f5b3b2db`

Vendored files:
- `include/ggml.h`
- `include/ggml-cpu.h`
- `src/ggml-common.h`
- `src/ggml-impl.h`
- `src/ggml-quants.h`
- `src/ggml-quants.c`
- `src/ggml-cpu/ggml-cpu-impl.h`

These files are intentionally limited to the quantization code path needed by
`gguf_kernels.ggml_ref`. The repo-local shim in
`gguf_kernels/csrc/ggml_ref_shim.c` provides row-size, init, and dispatch logic
so we do not need to vendor `ggml.c` or the full backend stack.
