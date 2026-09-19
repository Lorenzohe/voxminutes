# VoxMinutes CUDA preset for GeForce GTX 1660 / GTX 1660 SUPER.
#
# These Turing cards expose compute capability 7.5 but do not have Tensor
# Cores. llama.cpp recommends forcing its quantized MMQ kernels instead of the
# FP16 cuBLAS path on this GPU family.
set(GGML_CUDA_FORCE_MMQ ON CACHE BOOL "Force quantized MMQ kernels for GTX 1660-series GPUs" FORCE)
set(GGML_CUDA_FORCE_CUBLAS OFF CACHE BOOL "Do not force cuBLAS on GTX 1660-series GPUs" FORCE)
message(STATUS "VoxMinutes GTX 1660 preset: GGML_CUDA_FORCE_MMQ=ON")
