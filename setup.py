# SPDX-License-Identifier: Apache-2.0

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
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
        "build_ext": BuildExtension,
    },
)
