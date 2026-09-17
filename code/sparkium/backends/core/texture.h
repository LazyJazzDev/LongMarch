#pragma once
// Portable port of SampleTexture (code/sparkium/shaders/bindings.hlsli).
//
// The graphics backends sample sdr/hdr texture arrays with `samplers[0]`, which
// Scene creates as a linear filter with the default (repeat) address mode, and
// flip v. The offline backends reproduce linear + repeat sampling directly.

#include "sparkium/backends/core/structs.h"

namespace sparkium::backends {

SPARKIUM_HD inline float4 SampleTexture(const DeviceScene &scene, int32_t texture_index, const float2 &uv) {
  if (texture_index < 0)
    return float4(1.0f, 1.0f, 1.0f, 1.0f);
  const TextureData &texture = scene.textures[texture_index & 0xFFFFFF];
  const float *pixels = scene.texture_pixels + texture.offset;

  float2 corrected = float2(uv.x, 1.0f - uv.y);
  int32_t width = static_cast<int32_t>(texture.width);
  int32_t height = static_cast<int32_t>(texture.height);
  float x = corrected.x * static_cast<float>(width) - 0.5f;
  float y = corrected.y * static_cast<float>(height) - 0.5f;
  float x0f = floorf(x), y0f = floorf(y);
  float fx = x - x0f, fy = y - y0f;
  int32_t x0 = static_cast<int32_t>(x0f), y0 = static_cast<int32_t>(y0f);

  auto wrap = [](int32_t value, int32_t size) {
    int32_t result = value % size;
    return result < 0 ? result + size : result;
  };
  int32_t x1 = wrap(x0 + 1, width), y1 = wrap(y0 + 1, height);
  x0 = wrap(x0, width);
  y0 = wrap(y0, height);

  auto fetch = [&](int32_t px, int32_t py) {
    const float *texel = pixels + (static_cast<size_t>(py) * width + px) * 4;
    return float4(texel[0], texel[1], texel[2], texel[3]);
  };
  float4 t00 = fetch(x0, y0), t10 = fetch(x1, y0), t01 = fetch(x0, y1), t11 = fetch(x1, y1);
  float4 top = device::lerp(t00, t10, fx);
  float4 bottom = device::lerp(t01, t11, fx);
  return device::lerp(top, bottom, fy);
}

}  // namespace sparkium::backends
