from __future__ import annotations

import pytest

from accuracy_utils import (
    ErrorThresholds,
    build_roundtrip_cases,
    compute_metrics,
    quantize_reference_tensor,
)
import gguf_np


ROUNDTRIP_QTYPES = (
    gguf_np.GGMLQuantizationType.F32,
    gguf_np.GGMLQuantizationType.F16,
    gguf_np.GGMLQuantizationType.BF16,
    gguf_np.GGMLQuantizationType.Q1_0,
    gguf_np.GGMLQuantizationType.Q4_0,
    gguf_np.GGMLQuantizationType.Q4_1,
    gguf_np.GGMLQuantizationType.Q5_0,
    gguf_np.GGMLQuantizationType.Q5_1,
    gguf_np.GGMLQuantizationType.Q8_0,
    gguf_np.GGMLQuantizationType.TQ1_0,
    gguf_np.GGMLQuantizationType.TQ2_0,
    gguf_np.GGMLQuantizationType.MXFP4,
    gguf_np.GGMLQuantizationType.NVFP4,
)

ERROR_THRESHOLDS = {
    gguf_np.GGMLQuantizationType.F32: ErrorThresholds(0.0, 0.0, 0.0, 0.999_999),
    gguf_np.GGMLQuantizationType.F16: ErrorThresholds(0.0, 0.0, 0.0, 0.999_999),
    gguf_np.GGMLQuantizationType.BF16: ErrorThresholds(1.0e-4, 3.0e-4, 2.0e-3, 0.999_99),
    gguf_np.GGMLQuantizationType.Q1_0: ErrorThresholds(2.4e-2, 5.2e-2, 5.2e-1, 0.86),
    gguf_np.GGMLQuantizationType.Q4_0: ErrorThresholds(3.5e-3, 1.4e-2, 7.0e-2, 0.997),
    gguf_np.GGMLQuantizationType.Q4_1: ErrorThresholds(3.5e-3, 7.5e-3, 6.5e-2, 0.998),
    gguf_np.GGMLQuantizationType.Q5_0: ErrorThresholds(1.8e-3, 7.0e-3, 3.5e-2, 0.999_4),
    gguf_np.GGMLQuantizationType.Q5_1: ErrorThresholds(1.6e-3, 3.5e-3, 3.2e-2, 0.999_5),
    gguf_np.GGMLQuantizationType.Q8_0: ErrorThresholds(2.2e-4, 4.5e-4, 4.5e-3, 0.999_98),
    gguf_np.GGMLQuantizationType.TQ1_0: ErrorThresholds(1.7e-2, 3.2e-2, 5.2e-1, 0.91),
    gguf_np.GGMLQuantizationType.TQ2_0: ErrorThresholds(1.7e-2, 3.2e-2, 5.2e-1, 0.91),
    gguf_np.GGMLQuantizationType.MXFP4: ErrorThresholds(5.0e-3, 1.7e-2, 1.5e-1, 0.991),
    gguf_np.GGMLQuantizationType.NVFP4: ErrorThresholds(4.2e-3, 1.5e-2, 1.2e-1, 0.993),
}
@pytest.mark.parametrize(("qtype", "tensor_name"), build_roundtrip_cases(ROUNDTRIP_QTYPES))
def test_quantize_dequantize_matches_float16_reference(
    qtype: gguf_np.GGMLQuantizationType, tensor_name: str
) -> None:
    reference_fp16, quantized = quantize_reference_tensor(qtype, tensor_name)
    reference = reference_fp16.astype("float32", copy=False)
    dequantized = gguf_np.dequantize(quantized, qtype)

    assert dequantized.shape == reference.shape
    assert str(dequantized.dtype) == "float32"

    metrics = compute_metrics(reference, dequantized)
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
