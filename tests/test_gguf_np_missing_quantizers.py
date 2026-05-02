from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gguf_np

try:
    import libgguf
except ImportError:
    libgguf = None


MISSING_QTYPES = (
    gguf_np.GGMLQuantizationType.Q2_K,
    gguf_np.GGMLQuantizationType.Q3_K,
    gguf_np.GGMLQuantizationType.Q4_K,
    gguf_np.GGMLQuantizationType.Q5_K,
    gguf_np.GGMLQuantizationType.Q6_K,
    gguf_np.GGMLQuantizationType.IQ2_XXS,
    gguf_np.GGMLQuantizationType.IQ2_XS,
    gguf_np.GGMLQuantizationType.IQ2_S,
    gguf_np.GGMLQuantizationType.IQ3_XXS,
    gguf_np.GGMLQuantizationType.IQ3_S,
    gguf_np.GGMLQuantizationType.IQ1_S,
    gguf_np.GGMLQuantizationType.IQ1_M,
    gguf_np.GGMLQuantizationType.IQ4_NL,
    gguf_np.GGMLQuantizationType.IQ4_XS,
)


@pytest.mark.skipif(libgguf is None, reason="libgguf package is not available")
@pytest.mark.parametrize("qtype", MISSING_QTYPES)
def test_missing_quantizers_match_libgguf_reference(qtype: gguf_np.GGMLQuantizationType) -> None:
    block_size, _ = gguf_np.GGML_QUANT_SIZES[qtype]
    rows = np.linspace(-1.5, 1.5, 3 * block_size, dtype=np.float32).reshape(3, block_size)

    expected = libgguf.quantize_rows(rows, qtype)
    quantized = gguf_np.quantize(rows, qtype)
    dequantized = gguf_np.dequantize(quantized, qtype)

    assert np.array_equal(quantized, expected)
    assert quantized.shape == gguf_np.quant_shape_to_byte_shape(rows.shape, qtype)
    assert dequantized.shape == rows.shape
    assert dequantized.dtype == np.float32
    assert np.isfinite(dequantized).all()
