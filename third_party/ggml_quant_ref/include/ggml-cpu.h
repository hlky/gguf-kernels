#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Minimal vendored stub for ggml-quants.c.
 *
 * The quantization reference path in this repo only compiles ggml-quants.c and
 * a local dispatcher shim, so none of the CPU backend or allocator interfaces
 * from upstream ggml-cpu.h are required here.
 */

#ifdef __cplusplus
}
#endif
