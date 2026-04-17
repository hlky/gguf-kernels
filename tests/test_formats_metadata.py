from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gguf_kernels.formats import (
    GGML_FORMAT_INFO,
    GGML_QUANT_SIZES,
    GGMLQuantizationType,
    quant_shape_from_byte_shape,
    quant_shape_to_byte_shape,
)


def test_upstream_new_formats_are_registered() -> None:
    assert GGMLQuantizationType.NVFP4 in GGML_FORMAT_INFO
    assert GGMLQuantizationType.Q1_0 in GGML_FORMAT_INFO

    assert GGML_QUANT_SIZES[GGMLQuantizationType.NVFP4] == (64, 36)
    assert GGML_QUANT_SIZES[GGMLQuantizationType.Q1_0] == (128, 18)


def test_quant_shape_helpers_match_registered_sizes() -> None:
    assert quant_shape_to_byte_shape((3, 128), GGMLQuantizationType.Q1_0) == (3, 18)
    assert quant_shape_from_byte_shape((3, 18), GGMLQuantizationType.Q1_0) == (3, 128)

    assert quant_shape_to_byte_shape((2, 64), GGMLQuantizationType.NVFP4) == (2, 36)
    assert quant_shape_from_byte_shape((2, 36), GGMLQuantizationType.NVFP4) == (2, 64)
