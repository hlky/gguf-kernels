from __future__ import annotations

import pytest
import torch

from accuracy_utils import build_roundtrip_cases, quantize_reference_tensor

import gguf_np
import gguf_pt


NATIVE_QTYPES = tuple(
    qtype
    for qtype in sorted(gguf_pt.dequantize_functions, key=lambda q: q.value)
    if qtype
    not in (
        gguf_np.GGMLQuantizationType.BF16,
        gguf_np.GGMLQuantizationType.Q1_0,
        gguf_np.GGMLQuantizationType.Q4_0,
        gguf_np.GGMLQuantizationType.Q4_1,
        gguf_np.GGMLQuantizationType.Q5_0,
        gguf_np.GGMLQuantizationType.Q5_1,
        gguf_np.GGMLQuantizationType.Q8_0,
        gguf_np.GGMLQuantizationType.NVFP4,
    )
)

FALLBACK_QTYPES = tuple(
    qtype
    for qtype in sorted(gguf_np._type_traits, key=lambda q: q.value)
    if qtype not in gguf_pt.dequantize_functions
)


def dequantize_tensor_with_metadata(
    quantized: torch.Tensor,
    qtype: gguf_np.GGMLQuantizationType,
    shape: tuple[int, ...],
) -> torch.Tensor:
    quantized.tensor_type = qtype
    quantized.tensor_shape = shape
    return gguf_pt.dequantize_tensor(quantized, dtype=torch.float16)


def assert_matches_numpy_reference(
    dequantized: torch.Tensor,
    quantized: torch.Tensor,
    qtype: gguf_np.GGMLQuantizationType,
) -> None:
    expected = torch.from_numpy(
        gguf_np.dequantize(quantized.cpu().numpy(), qtype)
    ).to(torch.float16)
    diff = (dequantized.cpu() - expected).abs().to(torch.float32)

    assert dequantized.shape == expected.shape
    assert dequantized.dtype == torch.float16
    assert float(diff.mean()) <= 2.0e-5, (
        f"{qtype.name} mean_abs={float(diff.mean()):.6g} exceeded 2e-5"
    )
    assert float(diff.max()) <= 1.5e-4, (
        f"{qtype.name} max_abs={float(diff.max()):.6g} exceeded 1.5e-4"
    )


@pytest.mark.parametrize(("qtype", "tensor_name"), build_roundtrip_cases(NATIVE_QTYPES))
def test_torch_native_missing_dequantizers_match_numpy(
    qtype: gguf_np.GGMLQuantizationType, tensor_name: str
) -> None:
    reference_fp16, quantized = quantize_reference_tensor(qtype, tensor_name)
    quantized_torch = torch.from_numpy(quantized)

    dequantized = dequantize_tensor_with_metadata(
        quantized_torch, qtype, reference_fp16.shape
    )

    assert_matches_numpy_reference(dequantized, quantized_torch, qtype)


@pytest.mark.parametrize(("qtype", "tensor_name"), build_roundtrip_cases(FALLBACK_QTYPES))
def test_torch_fallback_missing_dequantizers_match_numpy(
    qtype: gguf_np.GGMLQuantizationType, tensor_name: str
) -> None:
    reference_fp16, quantized = quantize_reference_tensor(qtype, tensor_name)
    quantized_torch = torch.from_numpy(quantized)

    dequantized = dequantize_tensor_with_metadata(
        quantized_torch, qtype, reference_fp16.shape
    )

    assert_matches_numpy_reference(dequantized, quantized_torch, qtype)
