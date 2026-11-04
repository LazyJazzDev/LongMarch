#include "grassland/rhi/backend/backend_cuda_buffer.h"
#if defined(LONGMARCH_CUDA_RUNTIME)
#include <cstring>

namespace grassland::rhi::backend {
bool CheckCUDA(cudaError_t error, const char *operation) {
  if (error == cudaSuccess)
    return true;
  ReportError(static_cast<int32_t>(error), std::string(operation) + ": " + cudaGetErrorString(error));
  return false;
}

CUDADeviceScope::CUDADeviceScope(int device) {
  if (device < 0 || !CheckCUDA(cudaGetDevice(&previous_), "Get CUDA device"))
    return;
  changed_ = previous_ != device;
  valid_ = !changed_ || CheckCUDA(cudaSetDevice(device), "Select CUDA device");
}

CUDADeviceScope::~CUDADeviceScope() {
  if (valid_ && changed_)
    CheckCUDA(cudaSetDevice(previous_), "Restore CUDA device");
}

CUDADeviceScope::operator bool() const {
  return valid_;
}

CUDAMapping::CUDAMapping(int device) : device_(device) {
}

CUDAMapping::~CUDAMapping() {
  Reset();
}

void CUDAMapping::Reset() {
  if (!pointer_ && !memory_)
    return;
  CUDADeviceScope scope(device_);
  if (!scope)
    return;
  if (pointer_)
    CheckCUDA(cudaFree(pointer_), "Free CUDA mapping");
  if (memory_)
    CheckCUDA(cudaDestroyExternalMemory(memory_), "Destroy CUDA external memory");
  pointer_ = nullptr;
  memory_ = nullptr;
}

bool CUDAMapping::Import(const cudaExternalMemoryHandleDesc &desc, uint64_t size) {
  CUDADeviceScope scope(device_);
  if (!scope || !CheckCUDA(cudaImportExternalMemory(&memory_, &desc), "Import CUDA memory"))
    return false;
  cudaExternalMemoryBufferDesc buffer{};
  buffer.size = size;
  return CheckCUDA(cudaExternalMemoryGetMappedBuffer(&pointer_, memory_, &buffer), "Map CUDA buffer");
}

void *CUDAMapping::Pointer() const {
  return pointer_;
}

bool CUDAMapping::HasMemory() const {
  return memory_ != nullptr;
}

int FindCUDADeviceByUUID(const void *uuid) {
  int count{};
  if (cudaGetDeviceCount(&count) != cudaSuccess)
    return -1;
  for (int i = 0; i < count; ++i) {
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, i) == cudaSuccess &&
        std::memcmp(properties.uuid.bytes, uuid, sizeof(properties.uuid.bytes)) == 0)
      return i;
  }
  return -1;
}
#if defined(_WIN32)
int FindCUDADeviceByLUID(const void *luid) {
  int count{};
  if (cudaGetDeviceCount(&count) != cudaSuccess)
    return -1;
  for (int i = 0; i < count; ++i) {
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, i) == cudaSuccess && properties.luidDeviceNodeMask == 1 &&
        std::memcmp(properties.luid, luid, sizeof(properties.luid)) == 0)
      return i;
  }
  return -1;
}
#endif
}  // namespace grassland::rhi::backend
#endif
