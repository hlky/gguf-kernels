# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

from scripts.build_ggml_ref_lib import build_shared_lib, default_output_path


ROOT = Path(__file__).resolve().parent


def build_vendored_ggml_ref() -> Path:
    output = default_output_path()
    build_dir = ROOT / "build" / "ggml_ref"
    return build_shared_lib(output=output, build_dir=build_dir)


def copy_vendored_ggml_ref(build_lib: str | None, built: Path | None = None) -> None:
    if not build_lib:
        return

    built = built or build_vendored_ggml_ref()
    target_dir = Path(build_lib) / "gguf_kernels"
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(built, target_dir / built.name)


class BuildPyWithVendoredGGML(build_py):
    def run(self) -> None:
        built = build_vendored_ggml_ref()
        super().run()
        copy_vendored_ggml_ref(self.build_lib, built=built)


class BuildExtensionWithVendoredGGML(BuildExtension):
    def run(self) -> None:
        built = build_vendored_ggml_ref()
        super().run()
        if not self.inplace:
            copy_vendored_ggml_ref(self.build_lib, built=built)


setup(
    package_data={"gguf_kernels": ["*.so", "*.dylib", "*.dll"]},
    ext_modules=[
        CUDAExtension(
            name="gguf_kernels._C_gguf",
            sources=[
                "gguf_kernels/csrc/torch_bindings.cpp",
                "gguf_kernels/csrc/kernel.cu",
            ],
            include_dirs=[
                "gguf_kernels/csrc",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "-std=c++17", "--use_fast_math"],
            },
        )
    ],
    cmdclass={
        "build_ext": BuildExtensionWithVendoredGGML,
        "build_py": BuildPyWithVendoredGGML,
    },
)
