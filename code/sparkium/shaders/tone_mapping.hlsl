#include "native_contract.hlsli"
#include "tone_mapping.hlsli"
SP_RESOURCE(SP_TEXTURE(float4), accumulated_color, t0, 0);
#define SP_BINDING_accumulated_color SP_RESOURCE_ACCESS(SP_TEXTURE(float4), accumulated_color, 0)
SP_IMAGE_FORMAT("rgba8")
SP_RESOURCE(SP_RW_TEXTURE(float4), output, u0, 1);
#define SP_BINDING_output SP_RESOURCE_ACCESS(SP_RW_TEXTURE(float4), output, 1)

struct ToneMappingSettings {
  int view_transform;
  float exposure;
  float gamma;
  float contrast;
};

SP_RESOURCE(ConstantBuffer<ToneMappingSettings>, settings, b0, 2);
#define SP_BINDING_settings SP_RESOURCE_ACCESS(ConstantBuffer<ToneMappingSettings>, settings, 2)

float3 FilmicCurve(float3 color) {
  // Smooth shoulder/toe approximation for legacy Blender Filmic scenes.
  color = max(color, 0.0f);
  return saturate((color * (2.51f * color + 0.03f)) / (color * (2.43f * color + 0.59f) + 0.14f));
}

SP_NUMTHREADS(8, 8, 1) void Main(SP_CONTEXT uint3 dispatch_thread_id : SV_DispatchThreadID) {
  // Get the pixel coordinates
  uint2 pixel_coords = dispatch_thread_id.xy;

  // edge check
  uint width, height;
  SP_BINDING_accumulated_color.GetDimensions(width, height);
  if (pixel_coords.x >= width || pixel_coords.y >= height) {
    return;  // Out of bounds
  }

  // Read the accumulated color
  float4 color = SP_BINDING_accumulated_color.Load(int3(pixel_coords, 0));
  float3 linear_color = max(color.xyz * exp2(SP_BINDING_settings.exposure), 0.0f);

  float3 mapped_color;
  if (SP_BINDING_settings.view_transform == 1) {
    mapped_color = saturate(Linear2sRGB(linear_color));
  } else if (SP_BINDING_settings.view_transform == 2) {
    mapped_color = FilmicCurve(linear_color);
    mapped_color = saturate((mapped_color - 0.18f) * SP_BINDING_settings.contrast + 0.18f);
    mapped_color = pow(mapped_color, 1.0f / max(SP_BINDING_settings.gamma, 1.0e-4f));
  } else {
    float max_channel = max(linear_color.x, max(linear_color.y, linear_color.z));
    linear_color /= max(1.0f, max_channel);
    mapped_color = Linear2sRGB(linear_color);
  }

  // Write the result to the output image
  SP_BINDING_output[pixel_coords] = float4(mapped_color, color.w);
}
