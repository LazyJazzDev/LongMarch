#include <cuda_runtime.h>

#include <cstdint>

namespace {
__global__ void TransformValues(uint32_t *values, uint32_t count) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count)
    values[index] = values[index] * 3u + 7u;
}
}  // namespace

cudaError_t TransformCUDABuffer(void *pointer, uint32_t count, cudaStream_t stream) {
  TransformValues<<<(count + 255) / 256, 256, 0, stream>>>(static_cast<uint32_t *>(pointer), count);
  return cudaGetLastError();
}
