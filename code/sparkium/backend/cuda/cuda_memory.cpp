#include "sparkium/backend/cuda/cuda_memory.h"

#include "sparkium/backend/cuda/cuda_util.h"

namespace sparkium::backend {
void *AllocateCudaMemory(size_t size) {
  CUdeviceptr pointer;
  CheckCUDA(cuMemAlloc(&pointer, size));
  try {
    CheckCUDA(cuMemsetD8(pointer, 0, size));
  } catch (...) {
    cuMemFree(pointer);
    throw;
  }
  return reinterpret_cast<void *>(pointer);
}

void FreeCudaMemory(void *data) noexcept {
  cuMemFree(reinterpret_cast<CUdeviceptr>(data));
}

void UploadCudaMemory(void *destination, const void *source, size_t size, size_t offset) {
  CheckCUDA(cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(destination) + offset, source, size));
}

void DownloadCudaMemory(void *destination, const void *source, size_t size, size_t offset) {
  CheckCUDA(cuMemcpyDtoH(destination, reinterpret_cast<CUdeviceptr>(source) + offset, size));
}
}  // namespace sparkium::backend
