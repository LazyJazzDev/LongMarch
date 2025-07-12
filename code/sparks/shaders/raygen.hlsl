#include "common.hlsli"

RWTexture2D<float4> accumulated_color : register(u0, space0);
RWTexture2D<int> accumulated_samples : register(u0, space1);

[shader("raygeneration")] void Main() {
  // get the pixel coordinates
  uint2 pixel_coords = DispatchRaysIndex().xy;
  uint2 image_size = DispatchRaysDimensions().xy;
  float2 uv = (((float2(pixel_coords) + 0.5) / float2(image_size) * 2.0) - 1.0) * float2(1, -1);
  accumulated_color[pixel_coords] += float4(cos(uv), 0.0, 1.0);
  accumulated_samples[pixel_coords] += 1;
}
