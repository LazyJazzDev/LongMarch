#include "nbody_cuda.h"

#define BLOCK_SIZE 128
#define GRID_SIZE ((NUM_PARTICLE + BLOCK_SIZE - 1) / BLOCK_SIZE)

__global__ void UpdateKernel(const glm::vec3 *positions,
                             glm::vec3 *positions_write,
                             glm::vec3 *velocities,
                             int n_particle,
                             float delta_t) {
  uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;
  glm::vec3 pos = positions[id];
  glm::vec3 vel = velocities[id];
  glm::vec3 accel{0.0f};
  extern __shared__ glm::vec3 shared_pos[];
  for (int j = 0; j < n_particle; j += blockDim.x) {
    if (j + threadIdx.x < n_particle) {
      shared_pos[threadIdx.x] = positions[threadIdx.x + j];
    }
    __syncthreads();
#pragma unroll 128
    for (int i = 0; i < blockDim.x; i++) {
      auto diff = pos - shared_pos[i];
      auto l = 0.00125f * 0.00125f;
      l += diff.x * diff.x;
      l += diff.y * diff.y;
      l += diff.z * diff.z;
      l = rsqrt(l);
      l = l * l * l * (-delta_t * GRAVITY_COE);
      accel.x += diff.x * l;
      accel.y += diff.y * l;
      accel.z += diff.z * l;
    }
    __syncthreads();
  }
  vel += accel;
  pos.x += vel.x * delta_t;
  pos.y += vel.y * delta_t;
  pos.z += vel.z * delta_t;
  positions_write[id] = pos;
  velocities[id] = vel;
}

void UpdateStep(glm::vec3 *positions, glm::vec3 *velocities, glm::vec3 *positions_new, int n_particles, float delta_t) {
  UpdateKernel<<<GRID_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(glm::vec3)>>>(positions, positions_new, velocities,
                                                                          n_particles, delta_t);
  cudaMemcpyAsync(positions, positions_new, sizeof(glm::vec3) * n_particles, cudaMemcpyDeviceToDevice);
}
