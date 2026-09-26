#include "global_uniform_object.hlsli"

ConstantBuffer<GlobalUniformObject> ubo : register(b0, space0);
RWTexture2D<float4> output_image : register(u0, space1);

struct PSInput {
  float4 position : SV_POSITION;
  [[vk::location(0)]] float2 uv : TEXCOORD0;
};

PSInput VSMain(uint vertex_index : SV_VertexID) {
  float2 vertices[6] = {float2(-1.0, -1.0), float2(1.0, -1.0), float2(-1.0, 1.0),
                        float2(-1.0, 1.0),  float2(1.0, -1.0), float2(1.0, 1.0)};
  float2 v = vertices[vertex_index];
  PSInput ps_input;
  ps_input.position = float4(v, 0.0, 1.0);
  ps_input.uv = v;
  return ps_input;
}

void PSMain(PSInput input) {
  float2 uv = input.uv * 0.5 + 0.5;
  uint2 image_size;
  output_image.GetDimensions(image_size.x, image_size.y);
  float4 color = output_image[uint2(uv * image_size)];
  if (!ubo.hdr) {
    // Particles accumulate linear radiance. Only SDR needs sRGB encoding;
    // HDR stays linear, including highlights above reference white.
    float3 radiance = saturate(color.rgb);
    color.rgb = float3(radiance.r <= 0.0031308 ? 12.92 * radiance.r : 1.055 * pow(radiance.r, 1.0 / 2.4) - 0.055,
                       radiance.g <= 0.0031308 ? 12.92 * radiance.g : 1.055 * pow(radiance.g, 1.0 / 2.4) - 0.055,
                       radiance.b <= 0.0031308 ? 12.92 * radiance.b : 1.055 * pow(radiance.b, 1.0 / 2.4) - 0.055);
  }
  output_image[uint2(uv * image_size)] = color;
}
