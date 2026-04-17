from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

import gguf_np
from gguf_kernels.formats import GGML_QUANT_SIZES
from gguf_kernels.ops import ggml_dequantize


CUDA_BENCH_QTYPES = (
    gguf_np.GGMLQuantizationType.Q4_0,
    gguf_np.GGMLQuantizationType.Q4_1,
    gguf_np.GGMLQuantizationType.Q5_0,
    gguf_np.GGMLQuantizationType.Q5_1,
    gguf_np.GGMLQuantizationType.Q8_0,
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


@dataclass(frozen=True)
class BenchmarkResult:
    qtype: gguf_np.GGMLQuantizationType
    shape: tuple[int, ...]
    rows: int
    cols: int
    quantized_bytes: int
    output_bytes: int
    mean_ms: float
    min_ms: float
    max_ms: float
    values_per_s: float
    effective_gbps: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2048, help="Number of rows to dequantize.")
    parser.add_argument(
        "--cols",
        type=int,
        default=4096,
        help="Number of values per row after dequantization.",
    )
    parser.add_argument(
        "--shapes",
        type=str,
        default="",
        help=(
            "Semicolon-separated tensor shapes such as "
            "'512x4096;16x64x8192;8x16x16x16384'. "
            "When provided, --rows and --cols are ignored."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "float32"),
        default="float16",
        help="Output dtype requested from the CUDA kernel.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=25,
        help="Warmup iterations before measurement.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="Timed iterations per quantization format.",
    )
    parser.add_argument(
        "--qtypes",
        type=str,
        default="all",
        help="Comma-separated GGML quantization names, or 'all'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for random input generation.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare the first CUDA result against gguf_np.dequantize.",
    )
    return parser.parse_args()


def parse_shapes(spec: str, rows: int, cols: int) -> tuple[tuple[int, ...], ...]:
    if not spec:
        return ((rows, cols),)

    shapes = []
    for raw_shape in spec.split(";"):
        raw_shape = raw_shape.strip()
        if not raw_shape:
            continue
        dims = tuple(int(part) for part in raw_shape.lower().split("x"))
        if len(dims) < 2:
            raise ValueError(f"Shape must have at least 2 dims: {raw_shape}")
        if any(dim <= 0 for dim in dims):
            raise ValueError(f"Shape dims must be positive: {raw_shape}")
        shapes.append(dims)

    if not shapes:
        raise ValueError("No shapes were parsed from --shapes")
    return tuple(shapes)


def resolve_qtypes(spec: str) -> tuple[gguf_np.GGMLQuantizationType, ...]:
    if spec == "all":
        return CUDA_BENCH_QTYPES

    names = [name.strip().upper() for name in spec.split(",") if name.strip()]
    if not names:
        raise ValueError("No quantization types were selected")

    resolved = []
    for name in names:
        try:
            qtype = gguf_np.GGMLQuantizationType[name]
        except KeyError as exc:
            raise ValueError(f"Unknown quantization type: {name}") from exc
        if qtype not in CUDA_BENCH_QTYPES:
            raise ValueError(f"{name} is not wired in gguf_kernels/csrc/dequantize.h")
        resolved.append(qtype)
    return tuple(resolved)


def make_reference(shape: tuple[int, ...], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape, dtype=np.float32).astype(np.float16)


def benchmark_qtype(
    reference: np.ndarray,
    shape: tuple[int, ...],
    qtype: gguf_np.GGMLQuantizationType,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    check: bool,
) -> BenchmarkResult:
    quantized = gguf_np.quantize(reference, qtype)
    quantized_cuda = torch.from_numpy(quantized).cuda()

    rows = int(np.prod(shape[:-1], dtype=np.int64))
    cols = shape[-1]
    for _ in range(warmup):
        out = ggml_dequantize(quantized_cuda, qtype.value, rows, cols, dtype)
        del out
    torch.cuda.synchronize()

    if check:
        expected = gguf_np.dequantize(quantized, qtype)
        actual = (
            ggml_dequantize(quantized_cuda, qtype.value, rows, cols, dtype)
            .cpu()
            .numpy()
            .reshape(shape)
            .astype(np.float32, copy=False)
        )
        max_abs = float(np.abs(actual - expected).max())
        print(
            f"check {qtype.name} shape={format_shape(shape)}: max_abs={max_abs:.6g}",
            file=sys.stderr,
        )

    durations_ms = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start_event.record()
        out = ggml_dequantize(quantized_cuda, qtype.value, rows, cols, dtype)
        end_event.record()
        end_event.synchronize()
        durations_ms.append(start_event.elapsed_time(end_event))
        del out

    quantized_bytes = quantized.nbytes
    output_bytes = rows * cols * torch.empty((), dtype=dtype).element_size()
    mean_ms = mean(durations_ms)
    total_bytes = quantized_bytes + output_bytes
    values_per_s = (rows * cols) / (mean_ms / 1000.0)
    effective_gbps = total_bytes / (mean_ms / 1000.0) / 1e9

    return BenchmarkResult(
        qtype=qtype,
        shape=shape,
        rows=rows,
        cols=cols,
        quantized_bytes=quantized_bytes,
        output_bytes=output_bytes,
        mean_ms=mean_ms,
        min_ms=min(durations_ms),
        max_ms=max(durations_ms),
        values_per_s=values_per_s,
        effective_gbps=effective_gbps,
    )


def format_shape(shape: tuple[int, ...]) -> str:
    return "x".join(str(dim) for dim in shape)


def print_header(
    args: argparse.Namespace,
    qtypes: tuple[gguf_np.GGMLQuantizationType, ...],
    shapes: tuple[tuple[int, ...], ...],
) -> None:
    device = torch.cuda.get_device_name(torch.cuda.current_device())
    print(
        f"device={device} dtype={args.dtype} "
        f"warmup={args.warmup} iters={args.iters}"
    )
    print("qtypes=" + ",".join(q.name for q in qtypes))
    print("shapes=" + ";".join(format_shape(shape) for shape in shapes))
    print(
        "shape                 qtype     rows    cols  quant_MB  out_MB  mean_ms  "
        "min_ms  max_ms  Mvals/s  GB/s"
    )


def print_result(result: BenchmarkResult) -> None:
    print(
        f"{format_shape(result.shape):<22}"
        f"{result.qtype.name:<9}"
        f"{result.rows:>8}"
        f"{result.cols:>8}"
        f"{result.quantized_bytes / 1e6:>10.2f}"
        f"{result.output_bytes / 1e6:>8.2f}"
        f"{result.mean_ms:>9.3f}"
        f"{result.min_ms:>8.3f}"
        f"{result.max_ms:>8.3f}"
        f"{result.values_per_s / 1e6:>9.1f}"
        f"{result.effective_gbps:>7.2f}"
    )


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")

    qtypes = resolve_qtypes(args.qtypes)
    dtype = getattr(torch, args.dtype)
    shapes = parse_shapes(args.shapes, args.rows, args.cols)

    for shape in shapes:
        for qtype in qtypes:
            block_size, _ = GGML_QUANT_SIZES[qtype]
            if shape[-1] % block_size != 0:
                raise SystemExit(
                    f"{qtype.name} requires the last dim of {format_shape(shape)} "
                    f"to be a multiple of block size {block_size}"
                )

    print_header(args, qtypes, shapes)
    start_time = time.perf_counter()
    for shape_idx, shape in enumerate(shapes):
        reference = make_reference(shape, args.seed + shape_idx)
        for qtype in qtypes:
            result = benchmark_qtype(
                reference=reference,
                shape=shape,
                qtype=qtype,
                dtype=dtype,
                warmup=args.warmup,
                iters=args.iters,
                check=args.check,
            )
            print_result(result)
    elapsed = time.perf_counter() - start_time
    print(f"total_elapsed_s={elapsed:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
