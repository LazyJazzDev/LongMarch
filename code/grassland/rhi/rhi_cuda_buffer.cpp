#include "grassland/rhi/rhi_cuda_buffer.h"

#if defined(LONGMARCH_CUDA_RUNTIME)
namespace grassland::rhi {
ICUDABuffer::~ICUDABuffer() = default;
}  // namespace grassland::rhi
#endif
