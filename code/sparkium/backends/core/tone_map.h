#pragma once
// Portable port of code/sparkium/shaders/tone_mapping.hlsli and of the
// film2img + tone_mapping resolve chain used by Film::Develop:
//   code/sparkium/shaders/film2img.hlsl     (accumulated color -> average)
//   code/sparkium/shaders/tone_mapping.hlsl (average -> display RGBA8)

#include "sparkium/backends/core/structs.h"

namespace sparkium::backends {

SPARKIUM_HD inline float3 Linear2sRGB(const float3 &linear_color) {
  float3 cutoff = device::step(float3(0.0031308f), linear_color);
  float3 lower = linear_color * 12.92f;
  float3 higher = 1.055f * device::pow(linear_color, 1.0f / 2.4f) - 0.055f;
  return device::lerp(lower, higher, cutoff);
}

// Smooth shoulder/toe approximation for legacy Blender Filmic scenes.
SPARKIUM_HD inline float3 FilmicCurve(float3 color) {
  color = device::max(color, float3(0.0f));
  return device::saturate((color * (2.51f * color + 0.03f)) / (color * (2.43f * color + 0.59f) + 0.14f));
}

struct ToneMappingSettings {
  int32_t view_transform;
  float exposure;
  float gamma;
  float contrast;
};

SPARKIUM_HD inline float3 ApplyToneMapping(const float3 &value, const ToneMappingSettings &settings) {
  float3 linear_color = device::max(value * exp2f(settings.exposure), float3(0.0f));

  float3 mapped_color;
  if (settings.view_transform == 1) {
    mapped_color = device::saturate(Linear2sRGB(linear_color));
  } else if (settings.view_transform == 2) {
    mapped_color = FilmicCurve(linear_color);
    mapped_color = device::saturate((mapped_color - 0.18f) * settings.contrast + 0.18f);
    mapped_color = device::pow(mapped_color, 1.0f / device::max(settings.gamma, 1.0e-4f));
  } else {
    float max_channel = device::max(linear_color.x, device::max(linear_color.y, linear_color.z));
    linear_color /= device::max(1.0f, max_channel);
    mapped_color = Linear2sRGB(linear_color);
  }
  return mapped_color;
}

// film2img.hlsl loads the R32F sample image into an `int`, so a fractional
// accumulated weight (any scene with persistence != 1) is truncated before the
// division. The offline backends reproduce that conversion, otherwise their
// resolve would be a few percent off for exactly those scenes.
SPARKIUM_HD inline int32_t FilmResolveSampleCount(float accumulated_samples) {
  // Clamp before the conversion: the online shader relies on the hardware
  // conversion, which is only defined inside the int32 range.
  return static_cast<int32_t>(device::min(accumulated_samples, 2147483520.0f));
}

// film2img: divide the accumulated color by the accumulated sample count.
SPARKIUM_HD inline float3 ResolveAccumulated(const float4 &accumulated_color, int32_t accumulated_samples) {
  if (accumulated_samples == 0)
    return float3(0.0f, 0.0f, 0.0f);
  return (accumulated_color / static_cast<float>(accumulated_samples)).xyz();
}

// film2img writes alpha 1.0 for pixels without samples (float4(0, 0, 0, 1)).
SPARKIUM_HD inline float ResolveAccumulatedAlpha(const float4 &accumulated_color, int32_t accumulated_samples) {
  if (accumulated_samples == 0)
    return 1.0f;
  return accumulated_color.w / static_cast<float>(accumulated_samples);
}

}  // namespace sparkium::backends
