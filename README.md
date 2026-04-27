Exploration and benchmarking of gguf cuda kernels, [torch gguf implementation](https://github.com/city96/ComfyUI-GGUF) and reference numpy implementation.

Also, minimal vendored code from llama.cpp to enable standalone quantizing of formats that are unsupported in torch/numpy implementations, this may be separated into a releasable library.

End goal: implementing gguf kernels in [DinoML](https://github.com/hlky/dinoml)
