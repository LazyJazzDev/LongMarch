#pragma once
#include "grassland/rhi/rhi_cuda_buffer.h"
#if defined(LONGMARCH_CUDA_RUNTIME)
#include <cuda_runtime_api.h>

#include "grassland/rhi/backend/backend_util.h"

namespace grassland::rhi::backend {
bool CheckCUDA(cudaError_t error, const char *operation);

class CUDADeviceScope {
 public:
  explicit CUDADeviceScope(int device);
  CUDADeviceScope(const CUDADeviceScope &) = delete;
  CUDADeviceScope &operator=(const CUDADeviceScope &) = delete;
  ~CUDADeviceScope();
  explicit operator bool() const;

 private:
  int previous_{-1};
  bool valid_{};
  bool changed_{};
};

class CUDAMapping {
 public:
  explicit CUDAMapping(int device);
  CUDAMapping(const CUDAMapping &) = delete;
  CUDAMapping &operator=(const CUDAMapping &) = delete;
  ~CUDAMapping();
  bool Import(const cudaExternalMemoryHandleDesc &desc, uint64_t size);
  void *Pointer() const;
  bool HasMemory() const;
  void Reset();

 private:
  int device_;
  cudaExternalMemory_t memory_{};
  void *pointer_{};
};

// Match a CUDA Runtime UUID or Windows adapter LUID without changing current device.
int FindCUDADeviceByUUID(const void *uuid);
#if defined(_WIN32)
int FindCUDADeviceByLUID(const void *luid);
#endif
}  // namespace grassland::rhi::backend
#endif
