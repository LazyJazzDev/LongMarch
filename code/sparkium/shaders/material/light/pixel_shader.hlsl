#include "../../native_contract.hlsli"

struct PSInput {
  float4 position : SV_POSITION;
  [[vk::location(0)]] float3 world_position : TEXCOORD0;
  [[vk::location(1)]] float3 world_normal : TEXCOORD1;
  [[vk::location(2)]] float2 tex_coord : TEXCOORD2;
};

struct PSOutput {
  float4 radiance : SV_TARGET0;
  float4 albedo_roughness : SV_TARGET1;
  float4 position_specular : SV_TARGET2;
  float4 normal_metallic : SV_TARGET3;
  int stencil : SV_TARGET4;
};

struct MaterialLight {
  float3 emission;
};

SP_RESOURCE(ByteAddressBuffer, material_data, t0, 2);
#define SP_BINDING_material_data SP_RESOURCE_ACCESS(ByteAddressBuffer, material_data, 2)

PSOutput PSMain(PSInput input) {
  PSOutput SP_BINDING_output;
  float3 geom_normal;
  // compute geometry normal from position derivatives
  float3 dp1 = ddx(input.world_position);
  float3 dp2 = ddy(input.world_position);
  geom_normal = normalize(cross(dp2, dp1));
  if (length(input.world_normal) < 0.001) {
    input.world_normal = geom_normal;
  }
  float3 N = normalize(input.world_normal);
  float3 emission = SP_BINDING_material_data.Load<MaterialLight>(0).emission;
  SP_BINDING_output.radiance = float4(emission, 0.0);
  SP_BINDING_output.albedo_roughness = float4(0.0, 0.0, 0.0, 1.0);
  SP_BINDING_output.position_specular = float4(input.world_position, 0.0);
  SP_BINDING_output.normal_metallic = float4(N * 0.5 + 0.5, 0.0);
  SP_BINDING_output.stencil = 0;
  return SP_BINDING_output;
}
