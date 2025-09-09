#include "cao_di/util/util_util.h"

#include "log.h"

namespace CD {

#if defined(CHANGZHENG_CUDA_RUNTIME)
void CUDAThrowIfFailed(cudaError_t error, const std::string &message) {
  if (error != cudaSuccess) {
    LogError("[CUDA. ({})] " + message, static_cast<int>(error));
    throw std::runtime_error("[CUDA] " + message);
  }
}
#endif

}  // namespace CD
