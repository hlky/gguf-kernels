from __future__ import annotations

import pytest
import torch

from accuracy_utils import build_roundtrip_cases, quantize_reference_tensor, quantized_row_shape

import gguf_np


if not torch.cuda.is_available():
    pytest.skip("CUDA is required for gguf_kernels kernel tests", allow_module_level=True)

gguf_kernels_ops = pytest.importorskip("gguf_kernels.ops")

if not hasattr(torch.ops, "_C_gguf") or not hasattr(torch.ops._C_gguf, "ggml_dequantize"):
    pytest.skip(
        "gguf_kernels extension is not built; run `python setup.py build_ext --inplace`",
        allow_module_level=True,
    )


ROUNDTRIP_QTYPES = (
    gguf_np.GGMLQuantizationType.Q2_K,
    gguf_np.GGMLQuantizationType.Q3_K,
    gguf_np.GGMLQuantizationType.Q4_K,
    gguf_np.GGMLQuantizationType.Q5_K,
    gguf_np.GGMLQuantizationType.Q6_K,
    gguf_np.GGMLQuantizationType.IQ2_XXS,
    gguf_np.GGMLQuantizationType.IQ2_XS,
    gguf_np.GGMLQuantizationType.IQ3_XXS,
    gguf_np.GGMLQuantizationType.IQ1_S,
    gguf_np.GGMLQuantizationType.IQ4_NL,
    gguf_np.GGMLQuantizationType.IQ3_S,
    gguf_np.GGMLQuantizationType.IQ2_S,
    gguf_np.GGMLQuantizationType.IQ4_XS,
    gguf_np.GGMLQuantizationType.IQ1_M,
)


@pytest.mark.parametrize(("qtype", "tensor_name"), build_roundtrip_cases(ROUNDTRIP_QTYPES))
def test_cuda_missing_dequantizers_match_numpy_reference(
    qtype: gguf_np.GGMLQuantizationType, tensor_name: str
) -> None:
    reference_fp16, quantized = quantize_reference_tensor(qtype, tensor_name)
    row_count, row_width = quantized_row_shape(reference_fp16.shape)

    quantized_cuda = torch.from_numpy(quantized).cuda()
    dequantized = gguf_kernels_ops.ggml_dequantize(
        quantized_cuda,
        qtype.value,
        row_count,
        row_width,
        torch.float16,
    )

    expected = torch.from_numpy(gguf_np.dequantize(quantized, qtype)).to(torch.float16)
    actual = dequantized.reshape(reference_fp16.shape).cpu()
    diff = (actual - expected).abs().to(torch.float32)

    assert dequantized.is_cuda
    assert dequantized.dtype == torch.float16
    assert actual.shape == expected.shape
    assert float(diff.mean()) <= 2.0e-5, (
        f"{qtype.name} mean_abs={float(diff.mean()):.6g} exceeded 2e-5 for {tensor_name}"
    )
    assert float(diff.max()) <= 1.5e-4, (
        f"{qtype.name} max_abs={float(diff.max()):.6g} exceeded 1.5e-4 for {tensor_name}"
    )
