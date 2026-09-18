#pragma once

// Device mirror of the flattened scene plus the kernel launcher.
//
// The CUDA backend runs the same `pipelines/native/shared/*` functions the CPU
// backend runs; only the memory the `SceneView` points at differs. This class
// owns that device memory and re-uploads a blob when its revision advances.

#include <cstdint>
#include <memory>
#include <vector>

#include "sparkium/pipelines/native/core/scene_data.h"

namespace sparkium::native {

class CudaRenderer {
 public:
  // True when the build contains CUDA kernels and a device is usable.
  static bool Available();

  CudaRenderer();
  ~CudaRenderer();

  // Runs `settings.samples_per_dispatch` samples per pixel on the device and
  // reads the accumulation planes back.
  void Render(const SceneData &scene_data,
              uint2 extent,
              std::vector<float4> &accumulated_color,
              std::vector<float> &accumulated_samples);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sparkium::native
