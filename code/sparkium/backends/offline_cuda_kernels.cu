// CUDA kernels of the offline backend.
//
// They call the very same portable core the CPU backend uses
// (sparkium/backends/core), so the two backends differ only in the execution
// unit, not in the shading math.

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include "sparkium/backends/core/integrator.h"
#include "sparkium/backends/core/tone_map.h"
#include "sparkium/backends/offline_cuda_kernels.h"

namespace {

using sparkium::backends::DeviceScene;
using sparkium::backends::RenderSettings;
using sparkium::backends::ToneMappingSettings;
// CUDA also declares ::float4/::float3 in the global namespace, so the portable
// vector types get unambiguous local names here.
using Vec3 = sparkium::backends::float3;
using Vec4 = sparkium::backends::float4;

// One thread per pixel; the kernel mirrors raygen.hlsl's RenderPixel.
__global__ void OfflinePathTraceKernel(const DeviceScene *scene,
                                      RenderSettings settings,
                                      Vec4 *accumulated_color,
                                      float *accumulated_samples,
                                      unsigned int width,
                                      unsigned int height) {
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;
  const size_t index = static_cast<size_t>(y) * width + x;
  sparkium::backends::RenderPixel(*scene, settings, x, y, width, height, accumulated_color + index,
                                  accumulated_samples + index);
}

// film2img + tone_mapping.hlsl, evaluated with the shared core helpers.
__global__ void OfflineToneMapKernel(const Vec4 *accumulated_color,
                                     const float *accumulated_samples,
                                     uint8_t *rgba8,
                                     ToneMappingSettings settings,
                                     unsigned int width,
                                     unsigned int height) {
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;
  const size_t index = static_cast<size_t>(y) * width + x;
  const Vec4 color = accumulated_color[index];
  const float samples = accumulated_samples[index];
  const int sample_count = sparkium::backends::FilmResolveSampleCount(samples);
  const Vec3 average = sparkium::backends::ResolveAccumulated(color, sample_count);
  const float alpha = sparkium::backends::ResolveAccumulatedAlpha(color, sample_count);
  const Vec3 mapped = sparkium::backends::ApplyToneMapping(average, settings);
  uint8_t *pixel = rgba8 + index * 4;
  pixel[0] = static_cast<uint8_t>(__float2uint_rn(fminf(fmaxf(mapped.x, 0.0f), 1.0f) * 255.0f));
  pixel[1] = static_cast<uint8_t>(__float2uint_rn(fminf(fmaxf(mapped.y, 0.0f), 1.0f) * 255.0f));
  pixel[2] = static_cast<uint8_t>(__float2uint_rn(fminf(fmaxf(mapped.z, 0.0f), 1.0f) * 255.0f));
  pixel[3] = static_cast<uint8_t>(__float2uint_rn(fminf(fmaxf(alpha, 0.0f), 1.0f) * 255.0f));
}

void CheckCuda(cudaError_t status, const char *what) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("CUDA error in ") + what + ": " + cudaGetErrorString(status));
}

dim3 GridFor(unsigned int width, unsigned int height, dim3 block) {
  return dim3((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, 1);
}

}  // namespace

extern "C" void SparkiumOfflineCudaPathTrace(const void *device_scene,
                                             const void *host_settings,
                                             void *device_color,
                                             void *device_samples,
                                             unsigned int width,
                                             unsigned int height) {
  RenderSettings settings{};
  std::memcpy(&settings, host_settings, sizeof(RenderSettings));
  const dim3 block(16, 16, 1);
  OfflinePathTraceKernel<<<GridFor(width, height, block), block>>>(
      static_cast<const DeviceScene *>(device_scene), settings, static_cast<Vec4 *>(device_color),
      static_cast<float *>(device_samples), width, height);
  CheckCuda(cudaGetLastError(), "OfflinePathTraceKernel launch");
}

extern "C" void SparkiumOfflineCudaToneMap(const void *device_color,
                                           const void *device_samples,
                                           void *device_rgba8,
                                           const void *host_settings,
                                           unsigned int width,
                                           unsigned int height) {
  ToneMappingSettings settings{};
  std::memcpy(&settings, host_settings, sizeof(ToneMappingSettings));
  const dim3 block(16, 16, 1);
  OfflineToneMapKernel<<<GridFor(width, height, block), block>>>(static_cast<const Vec4 *>(device_color),
                                                                 static_cast<const float *>(device_samples),
                                                                 static_cast<uint8_t *>(device_rgba8), settings,
                                                                 width, height);
  CheckCuda(cudaGetLastError(), "OfflineToneMapKernel launch");
}

extern "C" void SparkiumOfflineCudaSynchronize() {
  CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}
