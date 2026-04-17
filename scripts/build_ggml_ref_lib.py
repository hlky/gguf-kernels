from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from setuptools._distutils.ccompiler import new_compiler
from setuptools._distutils.sysconfig import customize_compiler


def build_shared_lib(output: Path, build_dir: Path) -> Path:
    root = Path(__file__).resolve().parents[1]
    vendor_root = root / "third_party" / "ggml_quant_ref"

    compiler = new_compiler()
    customize_compiler(compiler)

    include_dirs = [
        str(vendor_root / "include"),
        str(vendor_root / "src"),
        str(vendor_root / "src" / "ggml-cpu"),
    ]

    compile_args = []
    link_args = []
    if compiler.compiler_type == "msvc":
        compile_args.extend(["/O2", "/std:c11"])
    else:
        compile_args.extend(["-O3", "-std=c11", "-fPIC"])
        link_args.extend(["-lm"])

    sources = [
        str(root / "gguf_kernels" / "csrc" / "ggml_ref_shim.c"),
        str(vendor_root / "src" / "ggml-quants.c"),
    ]

    build_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    objects = compiler.compile(
        sources=sources,
        output_dir=str(build_dir),
        include_dirs=include_dirs,
        macros=[("NDEBUG", "1")],
        extra_postargs=compile_args,
    )
    compiler.link_shared_object(
        objects=objects,
        output_filename=str(output),
        extra_postargs=link_args,
    )
    return output


def default_output_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    compiler = new_compiler()
    lib_name = compiler.library_filename("gguf_ggml_ref", lib_type="shared")
    return root / "gguf_kernels" / lib_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the vendored GGML quantization reference shared library")
    parser.add_argument("--output", type=Path, default=default_output_path(), help="Shared library output path")
    parser.add_argument("--build-dir", type=Path, default=Path("build/ggml_ref"), help="Object file build directory")
    parser.add_argument("--clean", action="store_true", help="Delete the build directory before compiling")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    build_dir = args.build_dir if args.build_dir.is_absolute() else root / args.build_dir
    output = args.output if args.output.is_absolute() else root / args.output

    if args.clean and build_dir.exists():
        shutil.rmtree(build_dir)

    built = build_shared_lib(output=output, build_dir=build_dir)
    print(built)


if __name__ == "__main__":
    main()
