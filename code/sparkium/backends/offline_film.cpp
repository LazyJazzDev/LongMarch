#include "sparkium/backends/offline_film.h"

#include <algorithm>

namespace sparkium::backends {

void OfflineFilm::Reset(uint32_t width, uint32_t height) {
  width_ = width;
  height_ = height;
  color_.assign(PixelCount(), device::float4(0.0f, 0.0f, 0.0f, 0.0f));
  samples_.assign(PixelCount(), 0.0f);
}

std::vector<uint8_t> OfflineFilm::Develop(const ToneMappingSettings &settings) const {
  return Develop(settings, color_.data(), samples_.data(), width_, height_);
}

std::vector<uint8_t> OfflineFilm::Develop(const ToneMappingSettings &settings,
                                          const device::float4 *color,
                                          const float *samples,
                                          uint32_t width,
                                          uint32_t height) {
  std::vector<uint8_t> rgba8(static_cast<size_t>(width) * height * 4);
  for (uint32_t y = 0; y < height; ++y) {
    for (uint32_t x = 0; x < width; ++x) {
      size_t index = static_cast<size_t>(y) * width + x;
      // film2img.hlsl: average the accumulated color, black when unsampled.
      const int32_t sample_count = FilmResolveSampleCount(samples[index]);
      float3 average = ResolveAccumulated(color[index], sample_count);
      float alpha = ResolveAccumulatedAlpha(color[index], sample_count);
      float3 mapped = ApplyToneMapping(average, settings);
      uint8_t *pixel = rgba8.data() + index * 4;
      // Round to nearest, ties to even: the IEEE conversion the GPU's UNORM
      // store and the CUDA backend's __float2uint_rn use.
      auto to_byte = [](float value) {
        return static_cast<uint8_t>(std::nearbyintf(device::clampf(value, 0.0f, 1.0f) * 255.0f));
      };
      pixel[0] = to_byte(mapped.x);
      pixel[1] = to_byte(mapped.y);
      pixel[2] = to_byte(mapped.z);
      pixel[3] = to_byte(alpha);
    }
  }
  return rgba8;
}

}  // namespace sparkium::backends
