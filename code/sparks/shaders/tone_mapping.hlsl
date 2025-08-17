#include "tone_mapping.hlsli"
Texture2D<float4> accumulated_color : register(t0, space0);
[[vk::image_format("rgba8")]] RWTexture2D<float4> output : register(u0, space1);

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

  // Apply tone mapping
  float3 mapped_color = Linear2sRGB(color.xyz, 2.2);

  // Write the result to the output image
  output[pixel_coords] = float4(mapped_color, color.w);
}
