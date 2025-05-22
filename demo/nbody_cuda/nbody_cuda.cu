#include "nbody_cuda.h"

#define BLOCK_SIZE 128
#define GRID_SIZE ((NUM_PARTICLE + BLOCK_SIZE - 1) / BLOCK_SIZE)

__global__ void UpdateKernel(const glm::vec4 *positions,
                             glm::vec4 *positions_write,
                             glm::vec4 *velocities,
                             int n_particle) {
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
      auto lsqr = 0.00125f * 0.00125f;
      lsqr += diff.x * diff.x;
      lsqr += diff.y * diff.y;
      lsqr += diff.z * diff.z;
      auto l = rsqrt(lsqr);
      lsqr = l * l * l * (-DELTA_T * GRAVITY_COE);
      accel.x += diff.x * lsqr;
      accel.y += diff.y * lsqr;
      accel.z += diff.z * lsqr;
    }
    __syncthreads();
  }
  vel += accel;
  pos.x += vel.x * DELTA_T;
  pos.y += vel.y * DELTA_T;
  pos.z += vel.z * DELTA_T;
  positions_write[id] = glm::vec4{pos, 1.0f};
  velocities[id] = glm::vec4{vel, 0.0f};
}

void UpdateStep(glm::vec4 *positions, glm::vec4 *velocities, int n_particles) {
  glm::vec4 *dev_positions;
  glm::vec4 *dev_velocities;
  glm::vec4 *dev_positions_write;
  cudaMalloc(&dev_positions, n_particles * sizeof(glm::vec4));
  cudaMalloc(&dev_velocities, n_particles * sizeof(glm::vec4));
  cudaMalloc(&dev_positions_write, n_particles * sizeof(glm::vec4));
  cudaMemcpy(dev_positions, positions, sizeof(glm::vec4) * n_particles, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_velocities, velocities, sizeof(glm::vec4) * n_particles, cudaMemcpyHostToDevice);

  UpdateKernel<<<GRID_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(glm::vec3)>>>(dev_positions, dev_positions_write,
                                                                          dev_velocities, n_particles);
  cudaDeviceSynchronize();

  cudaMemcpy(positions, dev_positions_write, sizeof(glm::vec4) * n_particles, cudaMemcpyDeviceToHost);
  cudaMemcpy(velocities, dev_velocities, sizeof(glm::vec4) * n_particles, cudaMemcpyDeviceToHost);
  cudaFree(dev_velocities);
  cudaFree(dev_positions);
  cudaFree(dev_positions_write);
}
