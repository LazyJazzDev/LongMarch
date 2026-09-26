#include "global_uniform_object.hlsli"

struct PSInput {
  float4 position : SV_POSITION;
  [[vk::location(0)]] float2 frag_v : TEXCOORD0;
};

ConstantBuffer<GlobalUniformObject> ubo : register(b0, space0);

PSInput VSMain([[vk::location(0)]] float3 pos
               : TEXCOORD0, uint vertex_index
               : SV_VertexID, uint instance_index
               : SV_InstanceID) {
  float2 vertices[6] = {float2(-1.0, -1.0), float2(1.0, -1.0), float2(-1.0, 1.0),
                        float2(-1.0, 1.0),  float2(1.0, -1.0), float2(1.0, 1.0)};
  float2 v = vertices[vertex_index];
  PSInput ps_input;
  ps_input.position =
      mul(ubo.world_to_screen, float4(pos, 1.0) + mul(ubo.camera_to_world, float4(v, 0.0, 0.0) * ubo.particle_size));
  ps_input.frag_v = v;
  return ps_input;
}

float4 PSMain(PSInput input) : SV_TARGET {
  float scale = max(1.0 - length(input.frag_v), 0.0);
  // Decode the original sRGB particle tint before linear accumulation.
  const float3 linear_tint = float3(1.0, 0.13286832, 0.03310477);
  // A compact, bright stellar core: half the radius and eight times the peak.
  const float peak_radiance = 8.0;
  return float4(linear_tint * peak_radiance * scale * scale * scale, 0.0);
}
