#include "tone_mapping.hlsli"
Texture2D<float4> accumulated_color : register(t0, space0);
#ifdef SPARKIUM_HDR_OUTPUT
[[vk::image_format("rgba32f")]] RWTexture2D<float4> output : register(u0, space1);
#else
[[vk::image_format("rgba8")]] RWTexture2D<float4> output : register(u0, space1);
#endif

struct ToneMappingSettings {
  int view_transform;
  float exposure;
  float gamma;
  float contrast;
};

ConstantBuffer<ToneMappingSettings> settings : register(b0, space2);

float3 FilmicCurve(float3 color) {
  // Smooth shoulder/toe approximation for legacy Blender Filmic scenes.
  color = max(color, 0.0f);
  return saturate((color * (2.51f * color + 0.03f)) / (color * (2.43f * color + 0.59f) + 0.14f));
}

[numthreads(8, 8, 1)] void Main(uint3 dispatch_thread_id
                                : SV_DispatchThreadID) {
  // Get the pixel coordinates
  uint2 pixel_coords = dispatch_thread_id.xy;

  // edge check
  uint width, height;
  accumulated_color.GetDimensions(width, height);
  if (pixel_coords.x >= width || pixel_coords.y >= height) {
    return;  // Out of bounds
  }

  // Read the accumulated color
  float4 color = accumulated_color.Load(int3(pixel_coords, 0));
  float3 linear_color = max(color.xyz * exp2(settings.exposure), 0.0f);

#ifdef SPARKIUM_HDR_OUTPUT
  // Preserve extended brightness and avoid sRGB encoding or SDR tone mapping.
  // Bound to the finite range of the RGBA16Float presentation surface.
  output[pixel_coords] = float4(min(linear_color, 65504.0f), color.w);
#else
  float3 mapped_color;
  if (settings.view_transform == 1) {
    mapped_color = saturate(Linear2sRGB(linear_color));
  } else if (settings.view_transform == 2) {
    mapped_color = FilmicCurve(linear_color);
    mapped_color = saturate((mapped_color - 0.18f) * settings.contrast + 0.18f);
    mapped_color = pow(mapped_color, 1.0f / max(settings.gamma, 1.0e-4f));
  } else {
    float max_channel = max(linear_color.x, max(linear_color.y, linear_color.z));
    linear_color /= max(1.0f, max_channel);
    mapped_color = Linear2sRGB(linear_color);
  }

  // Write the result to the output image
  output[pixel_coords] = float4(mapped_color, color.w);
#endif
}
