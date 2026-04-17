from __future__ import annotations

import pytest
import torch

from accuracy_utils import (
    ErrorThresholds,
    build_roundtrip_cases,
    compute_metrics,
    quantize_reference_tensor,
    quantized_row_shape,
)

import gguf_np

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for gguf_kernels accuracy tests", allow_module_level=True)

gguf_kernels_ops = pytest.importorskip("gguf_kernels.ops")

if not hasattr(torch.ops, "_C_gguf") or not hasattr(torch.ops._C_gguf, "ggml_dequantize"):
    pytest.skip(
        "gguf_kernels extension is not built; run `python setup.py build_ext --inplace`",
        allow_module_level=True,
    )


ROUNDTRIP_QTYPES = (
    gguf_np.GGMLQuantizationType.Q4_0,
    gguf_np.GGMLQuantizationType.Q4_1,
    gguf_np.GGMLQuantizationType.Q5_0,
    gguf_np.GGMLQuantizationType.Q5_1,
    gguf_np.GGMLQuantizationType.Q8_0,
)

ERROR_THRESHOLDS = {
    gguf_np.GGMLQuantizationType.Q4_0: ErrorThresholds(3.5e-3, 1.4e-2, 7.0e-2, 0.997),
    gguf_np.GGMLQuantizationType.Q4_1: ErrorThresholds(3.5e-3, 7.5e-3, 6.5e-2, 0.998),
    gguf_np.GGMLQuantizationType.Q5_0: ErrorThresholds(1.8e-3, 7.0e-3, 3.5e-2, 0.999_4),
    gguf_np.GGMLQuantizationType.Q5_1: ErrorThresholds(1.6e-3, 3.5e-3, 3.2e-2, 0.999_5),
    gguf_np.GGMLQuantizationType.Q8_0: ErrorThresholds(2.2e-4, 4.5e-4, 4.5e-3, 0.999_98),
}


@pytest.mark.parametrize(("qtype", "tensor_name"), build_roundtrip_cases(ROUNDTRIP_QTYPES))
def test_cuda_dequantize_matches_float16_reference(
    qtype: gguf_np.GGMLQuantizationType, tensor_name: str
) -> None:
    reference_fp16, quantized = quantize_reference_tensor(qtype, tensor_name)
    reference = reference_fp16.astype("float32", copy=False)
    row_count, row_width = quantized_row_shape(reference_fp16.shape)

    quantized_cuda = torch.from_numpy(quantized).cuda()
    dequantized = gguf_kernels_ops.ggml_dequantize(
        quantized_cuda,
        qtype.value,
        row_count,
        row_width,
        torch.float16,
    )

    assert dequantized.shape == (row_count, row_width)
    assert dequantized.dtype == torch.float16
    assert dequantized.is_cuda

    dequantized_np = (
        dequantized.reshape(reference_fp16.shape).cpu().numpy().astype("float32")
    )
    metrics = compute_metrics(reference, dequantized_np)
    limits = ERROR_THRESHOLDS[qtype]

    assert metrics["mean_abs"] <= limits.mean_abs, (
        f"{qtype.name} mean_abs={metrics['mean_abs']:.6g} "
        f"exceeded {limits.mean_abs:.6g} for {tensor_name}"
    )
    assert metrics["max_abs"] <= limits.max_abs, (
        f"{qtype.name} max_abs={metrics['max_abs']:.6g} "
        f"exceeded {limits.max_abs:.6g} for {tensor_name}"
    )
    assert metrics["nrmse"] <= limits.nrmse, (
        f"{qtype.name} nrmse={metrics['nrmse']:.6g} "
        f"exceeded {limits.nrmse:.6g} for {tensor_name}"
    )
    assert metrics["cosine"] >= limits.cosine_min, (
        f"{qtype.name} cosine={metrics['cosine']:.6g} "
        f"fell below {limits.cosine_min:.6g} for {tensor_name}"
    )
