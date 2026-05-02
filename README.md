Exploration and benchmarking of gguf cuda kernels, [torch gguf implementation](https://github.com/city96/ComfyUI-GGUF) and reference numpy implementation.

Also includes `libgguf/`, a separate Python package with CPython bindings for the minimized llama.cpp-derived reference quantizers used by `gguf_np` for formats that are not implemented directly in NumPy.

End goal: implementing gguf kernels in [DinoML](https://github.com/hlky/dinoml)
