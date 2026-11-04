#include "grassland/rhi/backend/backend_util.h"

#include "grassland/util/log.h"
#if defined(LONGMARCH_CUDA_RUNTIME)
#include <cuda_runtime_api.h>

#include <cstring>
#endif

namespace grassland::rhi::backend {
void ReportError(int32_t error_code, const std::string &error_message) {
  LogError("Error {}: {}", error_code, error_message);
}
#if defined(LONGMARCH_CUDA_RUNTIME)
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
#endif
}  // namespace grassland::rhi::backend
