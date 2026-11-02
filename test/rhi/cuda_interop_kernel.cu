#include <cuda_runtime.h>

#include <cstdint>

namespace {
__global__ void AddValues(uint32_t *values, uint32_t count, uint32_t increment) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count)
    values[index] += increment;
}
}  // namespace

cudaError_t AddCUDAInteropValues(void *pointer, uint32_t count, uint32_t increment, cudaStream_t stream) {
  AddValues<<<(count + 255) / 256, 256, 0, stream>>>(static_cast<uint32_t *>(pointer), count, increment);
  return cudaGetLastError();
}
