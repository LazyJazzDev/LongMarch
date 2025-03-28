#include "inverse.h"

struct LightInfo {
  float3 intensity;
};

[[vk::binding(0, 0)]] Texture2D<float4> g_albedo_texture : register(t0, space0);
[[vk::binding(0, 1)]] Texture2D<float4> g_position_texture : register(t0, space1);
[[vk::binding(0, 2)]] Texture2D<float4> g_normal_texture : register(t0, space2);
[[vk::binding(0, 3)]] Texture2D<float> g_depth_texture : register(t0, space3);
[[vk::binding(0, 4)]] ConstantBuffer<LightInfo> light_info : register(b0, space4);

struct PSInput {
  float4 position : SV_POSITION;
};

PSInput VSMain(uint vertex_index : SV_VertexID) {
  float2 rect_poses[6] = {
      float2(-1.0f, -1.0f), float2(1.0f, -1.0f), float2(-1.0f, 1.0f),
      float2(1.0f, -1.0f),  float2(1.0f, 1.0f),  float2(-1.0f, 1.0f),
  };
  PSInput output;
  output.position = float4(rect_poses[vertex_index], 0.0f, 1.0f);
  return output;
}

struct PSOutput {
  float4 exposure : SV_TARGET0;
};

PSOutput PSMain(PSInput input) {
  uint2 pixel_coord = uint2(input.position.xy);
  PSOutput output;
  float4 albedo = g_albedo_texture.Load(int3(pixel_coord, 0));
  output.exposure = float4(albedo.xyz * light_info.intensity, 0.0);
  return output;
}
