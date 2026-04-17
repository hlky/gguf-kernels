# Standalone GGML Quantization Library Plan

This note is based on a local clone of `llama.cpp` at commit `ed57d8e`.

## What To Treat As Source Of Truth

- `llama.cpp/ggml/src/ggml.c`
  - Canonical type table (`ggml_type_traits`)
  - Block sizes, type sizes, type names
  - Top-level quantization dispatch in `ggml_quantize_chunk`
- `llama.cpp/ggml/src/ggml-quants.h`
  - Public list of row/block quantize and dequantize functions
- `llama.cpp/ggml/src/ggml-quants.c`
  - Actual block codecs, format-specific helpers, and row validation
- `llama.cpp/src/llama-quant.cpp`
  - Model-level quantization policy, file-type presets, tensor fallbacks, and `imatrix` rules

The standalone library should port the first three layers first. `llama-quant.cpp` should be treated as an optional preset/policy layer, not as the core codec API.

## Format Inventory

### Dense / scalar storage

- `I8`
- `I16`
- `I32`
- `I64`
- `F64`
- `F32`
- `F16`
- `BF16`

### First-class quantized block formats

- `Q1_0`
- `Q4_0`
- `Q4_1`
- `Q5_0`
- `Q5_1`
- `Q8_0`
- `MXFP4`
- `NVFP4`
- `Q2_K`
- `Q3_K`
- `Q4_K`
- `Q5_K`
- `Q6_K`
- `TQ1_0`
- `TQ2_0`
- `IQ2_XXS`
- `IQ2_XS`
- `IQ2_S`
- `IQ3_XXS`
- `IQ3_S`
- `IQ1_S`
- `IQ1_M`
- `IQ4_NL`
- `IQ4_XS`

### Special / internal-looking formats

- `Q8_1`
  - Has block type and `quantize_row_q8_1_ref`
  - No public `dequantize_row_q8_1` declaration in `ggml-quants.h`
  - Not handled by `ggml_quantize_chunk`
- `Q8_K`
  - Has `quantize_row_q8_K_ref` and `dequantize_row_q8_K`
  - Present in validation
  - Not exposed as a `ggml_quantize_chunk` target

### Deprecated / removed runtime repacks

- `Q4_0_4_4`
- `Q4_0_4_8`
- `Q4_0_8_8`
- `IQ4_NL_4_4`
- `IQ4_NL_4_8`
- `IQ4_NL_8_8`

Recommendation: support these only as explicit decode/reject cases, not as normal encode targets.

## Important Design Observations

1. `ggml.c` and `ggml-quants.*` define concrete codecs. `llama-quant.cpp` defines presets and heuristics.
2. `llama_ftype` is not a 1:1 encoding of block formats.
   - Example: `LLAMA_FTYPE_MOSTLY_IQ2_S` maps to `GGML_TYPE_IQ2_XS` as the default starting point.
   - Example: `LLAMA_FTYPE_MOSTLY_IQ3_XS` maps to `GGML_TYPE_IQ3_S`.
3. `imatrix` handling exists at two levels.
   - Low-level codec dispatch in `ggml.c` only hard-requires it for a small subset.
   - `llama-quant.cpp` applies broader policy requirements based on tensor role and chosen preset.
4. Some formats need initialization and shared tables.
   - `IQ2_*`, `IQ1_*`, and `IQ3_*` use init/free paths and cached grids.
5. Validation is a real feature, not just test code.
   - `ggml_validate_row_data` should become part of the standalone API.

## Proposed Library Scope

### Core library

- Exact block layouts and constants
- Row size / block size / type metadata
- Encode and decode for every supported concrete format
- Row-data validation
- Optional init/free for formats with shared lookup tables
- Pure C API with stable ABI

### Optional layers on top

- C++ convenience wrappers
- Python bindings
- PyTorch bindings
- Preset engine that mirrors `llama_ftype` and tensor-category heuristics
- GGUF tensor read/write helpers

## Recommended Module Split

- `include/ggq/type.h`
  - enums, constants, public metadata structs
- `include/ggq/api.h`
  - stable C API
- `src/type_table.c`
  - port of `ggml_type_traits`, row-size helpers, type-name helpers
- `src/codec/*.c`
  - one family per file:
  - `q_basic.c` for `Q1_0`, `Q4_*`, `Q5_*`, `Q8_0`
  - `q_k.c` for `Q*_K`
  - `iq.c` for `IQ*`
  - `tq.c` for `TQ*`
  - `fp4.c` for `MXFP4`, `NVFP4`
  - `dense.c` for `F16`, `BF16`, `F32`, `F64`, integer passthrough
- `src/init.c`
  - lookup-table lifecycle for `IQ*`
- `src/validate.c`
  - port of `ggml_validate_row_data`
- `src/preset/llama_ftype.cpp`
  - optional preset translation layer copied from `llama-quant.cpp`

## API Shape

- `ggq_type_name(type)`
- `ggq_block_size(type)`
- `ggq_type_size(type)`
- `ggq_row_size(type, n_per_row)`
- `ggq_is_quantized(type)`
- `ggq_requires_init(type)`
- `ggq_requires_imatrix(type)`
- `ggq_init(type)`
- `ggq_free_all()`
- `ggq_encode(type, src_f32, dst, nrows, n_per_row, imatrix)`
- `ggq_decode(type, src, dst_f32, nrows, n_per_row)`
- `ggq_validate(type, data, nbytes)`

Keep the core API format-centric. Do not make tensor names, model arches, or layer heuristics part of the base library.

## Implementation Order

### Phase 1: metadata and decode completeness

- Port type metadata from `ggml.c`
- Port decode for all active first-class formats
- Add byte-for-byte layout tests and float-accuracy tests against `llama.cpp`
- Add explicit behavior for deprecated runtime-repack types

### Phase 2: encode completeness

- Port encode paths for straightforward formats:
  - `Q1_0`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`
  - `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`
  - `TQ1_0`, `TQ2_0`
  - `MXFP4`, `NVFP4`
- Port `IQ*` encode paths with lookup-table initialization
- Keep `Q8_1` and `Q8_K` out of the public encode surface unless there is a concrete use case

### Phase 3: policy and presets

- Add a separate preset module that mirrors:
  - `llama_ftype` to default type mapping
  - `imatrix` policy
  - tensor fallback logic
- Expose this as optional behavior, not core codec logic

### Phase 4: bindings and acceleration

- Re-point this repo's NumPy / Torch / CUDA code to the standalone core
- Keep CUDA kernels as an acceleration layer over a CPU-correct reference implementation

## Test Strategy

- Golden-layout tests:
  - compare packed bytes against `llama.cpp` reference output
- Round-trip tests:
  - `f32 -> quant -> f32`
- Cross-validation tests:
  - standalone decode vs `llama.cpp` decode
  - standalone encode vs `ggml_quantize_chunk`
- Edge-case tests:
  - wrong row size
  - missing `imatrix`
  - NaN / Inf validation
  - unsupported deprecated types
- Coverage matrix:
  - one test per active type
  - separate tests for `Q8_1` and `Q8_K` special handling

## Immediate Next Steps For This Repo

1. Create a canonical shared type/metadata table derived from `llama.cpp/ggml/src/ggml.c`.
2. Separate "format codec support" from "PyTorch/CUDA kernel support" in the current docs and API.
3. Fill decode gaps first so the repo can claim full read support.
4. Add a reference CPU validation harness that uses the local `llama.cpp` clone as the oracle.
5. Decide whether `Q8_1` and `Q8_K` are public targets or internal-only compatibility formats.
