#include "grassland/util/util_util.h"

#include "log.h"

namespace grassland {

#if defined(LONGMARCH_CUDA_RUNTIME)
void CUDAThrowIfFailed(cudaError_t error, const std::string &message) {
  if (error != cudaSuccess) {
    LogError("[CUDA. ({})] " + message, static_cast<int>(error));
    throw std::runtime_error("[CUDA] " + message);
  }
}
#endif

}  // namespace grassland
