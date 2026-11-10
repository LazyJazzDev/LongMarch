#pragma once
#include "grassland/rhi/rhi_cuda_buffer.h"
#if defined(LONGMARCH_CUDA_RUNTIME)
#include <cuda_runtime_api.h>

#include "grassland/rhi/backend/backend_util.h"

namespace grassland::rhi::backend {
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

}  // namespace grassland::rhi::backend
#endif
