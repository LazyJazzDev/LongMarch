#pragma once

#define NOMINMAX

#ifdef _WIN64
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include "pybind11/chrono.h"
#include "pybind11/eigen.h"
#include "pybind11/functional.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#if defined(LONGMARCH_CUDA_RUNTIME)
#include "cuda_runtime.h"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include "thrust/sort.h"
#endif

namespace grassland {

#if defined(LONGMARCH_CUDA_RUNTIME)
void CUDAThrowIfFailed(cudaError_t error, const std::string &message);
#endif

#if defined(__CUDACC__)
#define LM_DEVICE_FUNC __device__ __host__
#else
#define LM_DEVICE_FUNC
#endif

typedef enum DeviceType {
  CPU = 0,
  CUDA = 1,
} DeviceType;

struct VertexBufferView {
  const void *data;
  size_t stride;
  VertexBufferView() : data(nullptr), stride(0) {
  }
  VertexBufferView(const void *data_, size_t stride_, size_t offset_ = 0)
      : data(static_cast<const char *>(data_) + offset_), stride(stride_) {
  }

  template <typename T>
  VertexBufferView(const T *data_, size_t offset = 0) : data(data_), stride(sizeof(T)) {
    data = static_cast<const char *>(data) + offset;
  }

  template <typename T>
  LM_DEVICE_FUNC const T &Get(size_t index, size_t offset = 0) const {
    return *reinterpret_cast<const T *>(static_cast<const char *>(data) + stride * index + offset);
  }

  template <typename T>
  LM_DEVICE_FUNC T &At(size_t index, size_t offset = 0) {
    return *reinterpret_cast<T *>(static_cast<char *>(const_cast<void *>(data)) + stride * index + offset);
  }
};

}  // namespace grassland
