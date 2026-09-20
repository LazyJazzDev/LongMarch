#pragma once
#include <stdexcept>
#include <string>
#ifdef LONGMARCH_NATIVE_CUDA_ENABLED
#include <cuda.h>
#endif

namespace grassland::graphics::backend {

#ifdef LONGMARCH_NATIVE_CUDA_ENABLED
inline void CheckCUDA(CUresult result) {
  if (result != CUDA_SUCCESS) {
    const char *text = nullptr;
    cuGetErrorString(result, &text);
    throw std::runtime_error(std::string("native CUDA: ") + (text ? text : "unknown driver error"));
  }
}
#endif
[[noreturn]] inline void NativeUnsupported() {
  throw std::runtime_error(
      "operation unavailable on this headless native backend; use a graphics backend for rasterization or presentation");
}

}  // namespace grassland::graphics::backend
