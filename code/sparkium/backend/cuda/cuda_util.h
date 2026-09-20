#pragma once
#include <cuda.h>

#include <stdexcept>
#include <string>

namespace sparkium::backend {
inline void CheckCUDA(CUresult result) {
  if (result != CUDA_SUCCESS) {
    const char *text = nullptr;
    cuGetErrorString(result, &text);
    throw std::runtime_error(std::string("native CUDA: ") + (text ? text : "unknown driver error"));
  }
}
}  // namespace sparkium::backend
