#pragma once
// Host-side counterpart of raytracing::Film plus the film2img + tone_mapping
// resolve chain. The accumulated color/sample images keep the online formats
// (RGBA32F and R32F) so the same resolve code produces the same pixels.

#include <cstdint>
#include <vector>

#include "sparkium/backends/core/structs.h"
#include "sparkium/backends/core/tone_map.h"

namespace sparkium::backends {

class OfflineFilm {
 public:
  void Reset(uint32_t width, uint32_t height);

  uint32_t Width() const {
    return width_;
  }
  uint32_t Height() const {
    return height_;
  }
  size_t PixelCount() const {
    return static_cast<size_t>(width_) * height_;
  }

  device::float4 *Color() {
    return color_.data();
  }
  float *Samples() {
    return samples_.data();
  }
  const device::float4 *Color() const {
    return color_.data();
  }
  const float *Samples() const {
    return samples_.data();
  }

  // film2img then tone_mapping.hlsl, evaluated with the shared core code.
  std::vector<uint8_t> Develop(const ToneMappingSettings &settings) const;
  static std::vector<uint8_t> Develop(const ToneMappingSettings &settings,
                                      const device::float4 *color,
                                      const float *samples,
                                      uint32_t width,
                                      uint32_t height);

 private:
  uint32_t width_{0}, height_{0};
  std::vector<device::float4> color_;
  std::vector<float> samples_;
};

}  // namespace sparkium::backends
