#include <cuda_runtime.h>

#include <cstdint>

namespace {
constexpr unsigned block_size = 128;

__global__ void NBodyStep(const float3 *positions,
                          float3 *velocities,
                          float3 *next_positions,
                          uint32_t count,
                          float delta_t,
                          float gravity) {
  const unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  const float3 position = id < count ? positions[id] : make_float3(0, 0, 0);
  float3 acceleration = make_float3(0, 0, 0);
  __shared__ __align__(16) float3 tile[block_size];
  for (unsigned base = 0; base < count; base += block_size) {
    tile[threadIdx.x] = base + threadIdx.x < count ? positions[base + threadIdx.x] : make_float3(0, 0, 0);
    __syncthreads();
    const unsigned remaining = count - base < block_size ? count - base : block_size;
#pragma unroll 128
    for (unsigned i = 0; i < remaining; ++i) {
      const float3 difference = make_float3(position.x - tile[i].x, position.y - tile[i].y, position.z - tile[i].z);
      float inverse = rsqrtf(difference.x * difference.x + difference.y * difference.y + difference.z * difference.z +
                             0.00125f * 0.00125f);
      const float force = inverse * inverse * inverse * (-delta_t * gravity);
      acceleration.x += difference.x * force;
      acceleration.y += difference.y * force;
      acceleration.z += difference.z * force;
    }
    __syncthreads();
  }
  if (id < count) {
    float3 velocity = velocities[id];
    velocity.x += acceleration.x;
    velocity.y += acceleration.y;
    velocity.z += acceleration.z;
    velocities[id] = velocity;
    next_positions[id] = make_float3(position.x + velocity.x * delta_t, position.y + velocity.y * delta_t,
                                     position.z + velocity.z * delta_t);
  }
}
}  // namespace

cudaError_t RunNBodyCUDA(void *positions,
                         void *velocities,
                         void *next_positions,
                         uint32_t count,
                         float delta_t,
                         float gravity,
                         cudaStream_t stream) {
  if (!count)
    return cudaSuccess;
  NBodyStep<<<(count + block_size - 1) / block_size, block_size, 0, stream>>>(
      static_cast<float3 *>(positions), static_cast<float3 *>(velocities), static_cast<float3 *>(next_positions), count,
      delta_t, gravity);
  auto error = cudaGetLastError();
  if (error != cudaSuccess)
    return error;
  return cudaMemcpyAsync(positions, next_positions, uint64_t(count) * sizeof(float3), cudaMemcpyDeviceToDevice, stream);
}
