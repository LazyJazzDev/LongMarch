#pragma once
// Host-side entry points of the CUDA kernels that implement the offline
// backend's path tracing and film resolve. The declarations use plain C types
// so the host translation unit does not need nvcc.

#if defined(LONGMARCH_CUDA_ENABLED)

#include <cstdint>

extern "C" {

// Traces `settings.samples_per_dispatch` samples for every pixel of `color`
// (accumulated RGBA32F) and `samples` (accumulated R32F).
void SparkiumOfflineCudaPathTrace(const void *device_scene,
                                  const void *host_settings,
                                  void *device_color,
                                  void *device_samples,
                                  unsigned int width,
                                  unsigned int height);

// film2img + tone_mapping.hlsl for the whole image.
void SparkiumOfflineCudaToneMap(const void *device_color,
                                const void *device_samples,
                                void *device_rgba8,
                                const void *host_settings,
                                unsigned int width,
                                unsigned int height);

void SparkiumOfflineCudaSynchronize();
}

#endif  // LONGMARCH_CUDA_ENABLED
