#include <optional>

#include <torch/extension.h>
#include <torch/library.h>

torch::Tensor ggml_dequantize(torch::Tensor W, int64_t type, int64_t m,
                              int64_t n,
                              std::optional<at::ScalarType> const &dtype);

TORCH_LIBRARY(_C_gguf, ops)
{
    ops.def(
        "ggml_dequantize(Tensor W, int type, SymInt m, SymInt n, ScalarType? "
        "dtype) -> Tensor");
    ops.impl("ggml_dequantize", torch::kCUDA, &ggml_dequantize);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}