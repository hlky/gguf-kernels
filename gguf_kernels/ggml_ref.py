from __future__ import annotations

import ctypes
import ctypes.util
import os
import atexit
from functools import lru_cache
from pathlib import Path
from typing import Final

import numpy as np

from gguf_kernels.formats import GGMLQuantizationType, quant_shape_to_byte_shape


_C_FLOAT_P = ctypes.POINTER(ctypes.c_float)
_DLL_MODE = getattr(ctypes, "RTLD_GLOBAL", 0)
_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
_RELATIVE_LIB_PATHS: Final[tuple[str, ...]] = (
    "gguf_kernels/libgguf_ggml_ref.so",
    "gguf_kernels/libgguf_ggml_ref.dylib",
    "gguf_kernels/gguf_ggml_ref.dll",
    "build/ggml_ref/libgguf_ggml_ref.so",
    "build/ggml_ref/libgguf_ggml_ref.dylib",
    "build/ggml_ref/gguf_ggml_ref.dll",
    "build/src/libggml.so",
    "build/bin/libggml.so",
    "build/src/libggml.dylib",
    "build/bin/libggml.dylib",
    "build/bin/ggml.dll",
    "ggml/build/src/libggml.so",
    "ggml/build/bin/libggml.so",
    "llama.cpp/build/src/libggml.so",
    "llama.cpp/build/bin/libggml.so",
    ".cache/ggml-ref/build/src/libggml.so",
    ".cache/ggml-ref/build/bin/libggml.so",
)


def _candidate_library_refs() -> list[str]:
    refs: list[str] = []

    if env_path := os.environ.get("GGUF_GGML_LIB"):
        refs.append(env_path)

    for relative_path in _RELATIVE_LIB_PATHS:
        candidate = _REPO_ROOT / relative_path
        if candidate.is_file():
            refs.append(str(candidate))

    if system_ref := ctypes.util.find_library("ggml"):
        refs.append(system_ref)

    return refs


def _load_with_siblings(path: Path) -> ctypes.CDLL:
    for sibling_name in ("libggml-base.so", "libggml-cpu.so", "libggml-base.dylib", "libggml-cpu.dylib"):
        sibling = path.with_name(sibling_name)
        if sibling.is_file():
            ctypes.CDLL(str(sibling), mode=_DLL_MODE)
    return ctypes.CDLL(str(path), mode=_DLL_MODE)


def _configure_quant_api(lib: ctypes.CDLL) -> ctypes.CDLL:
    if hasattr(lib, "gguf_ggml_ref_quantize_chunk"):
        lib.gguf_ggml_ref_quantize_chunk.restype = ctypes.c_size_t
        lib.gguf_ggml_ref_quantize_chunk.argtypes = (
            ctypes.c_int,
            _C_FLOAT_P,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            _C_FLOAT_P,
        )
        lib.gguf_ggml_ref_quantize_requires_imatrix.restype = ctypes.c_bool
        lib.gguf_ggml_ref_quantize_requires_imatrix.argtypes = (ctypes.c_int,)
        lib._quantize_chunk = lib.gguf_ggml_ref_quantize_chunk
        lib._quantize_requires_imatrix = lib.gguf_ggml_ref_quantize_requires_imatrix
        if hasattr(lib, "gguf_ggml_ref_quantize_free"):
            lib.gguf_ggml_ref_quantize_free.restype = None
            lib.gguf_ggml_ref_quantize_free.argtypes = ()
            atexit.register(lib.gguf_ggml_ref_quantize_free)
        return lib

    if hasattr(lib, "ggml_quantize_chunk"):
        lib.ggml_quantize_chunk.restype = ctypes.c_size_t
        lib.ggml_quantize_chunk.argtypes = (
            ctypes.c_int,
            _C_FLOAT_P,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            _C_FLOAT_P,
        )
        lib.ggml_quantize_requires_imatrix.restype = ctypes.c_bool
        lib.ggml_quantize_requires_imatrix.argtypes = (ctypes.c_int,)
        lib._quantize_chunk = lib.ggml_quantize_chunk
        lib._quantize_requires_imatrix = lib.ggml_quantize_requires_imatrix
        if hasattr(lib, "ggml_quantize_free"):
            lib.ggml_quantize_free.restype = None
            lib.ggml_quantize_free.argtypes = ()
            atexit.register(lib.ggml_quantize_free)
        return lib

    raise AttributeError("missing GGML quantization entry points")


@lru_cache(maxsize=1)
def _load_ggml_lib() -> ctypes.CDLL:
    errors: list[str] = []

    for ref in _candidate_library_refs():
        try:
            lib = _load_with_siblings(Path(ref)) if Path(ref).is_file() else ctypes.CDLL(ref, mode=_DLL_MODE)
            return _configure_quant_api(lib)
        except OSError as exc:
            errors.append(f"{ref}: {exc}")
        except AttributeError as exc:
            errors.append(f"{ref}: missing ggml quantization symbols ({exc})")

    searched = ", ".join(_candidate_library_refs()) or "<none>"
    details = "; ".join(errors) if errors else "no candidate library was found"
    raise RuntimeError(
        "Unable to load libggml for GGML-backed quantization. "
        "Build the vendored shim with `python scripts/build_ggml_ref_lib.py` "
        f"or set GGUF_GGML_LIB to a compatible shared library. Searched: {searched}. {details}"
    )


def has_ggml_reference() -> bool:
    try:
        _load_ggml_lib()
    except RuntimeError:
        return False
    return True


def quantize_rows_with_ggml(data: np.ndarray, qtype: GGMLQuantizationType) -> np.ndarray:
    rows = np.ascontiguousarray(data, dtype=np.float32)
    if rows.ndim == 0:
        raise ValueError("Expected an array with at least one dimension")

    lib = _load_ggml_lib()

    out = np.zeros(quant_shape_to_byte_shape(rows.shape, qtype), dtype=np.uint8, order="C")
    n_rows = int(np.prod(rows.shape[:-1], dtype=np.int64)) if rows.ndim > 1 else 1

    if lib._quantize_requires_imatrix(qtype.value):
        # Match upstream gguf-py reference behavior for activation-aware formats.
        weights = np.sum((rows * rows).reshape((-1, rows.shape[-1])), axis=0, dtype=np.float32)
        weights_p = weights.ctypes.data_as(_C_FLOAT_P)
    else:
        weights_p = ctypes.cast(0, _C_FLOAT_P)

    out_size = lib._quantize_chunk(
        qtype.value,
        rows.ctypes.data_as(_C_FLOAT_P),
        out.ctypes.data_as(ctypes.c_void_p),
        0,
        n_rows,
        rows.shape[-1],
        weights_p,
    )
    if out_size != out.size:
        raise RuntimeError(f"ggml_quantize_chunk returned {out_size} bytes for {qtype.name}, expected {out.size}")

    return out
