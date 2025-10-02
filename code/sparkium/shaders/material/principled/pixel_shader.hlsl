#include "buffer_helper.hlsli"

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

ByteAddressBuffer material_data : register(t0, space2);

PSOutput PSMain(PSInput input) {
  PSOutput output;
  float3 geom_normal;
  // compute geometry normal from position derivatives
  float3 dp1 = ddx(input.world_position);
  float3 dp2 = ddy(input.world_position);
  geom_normal = normalize(cross(dp1, dp2));
  if (length(input.world_normal) < 0.001) {
    input.world_normal = geom_normal;
  }
  float3 N = normalize(input.world_normal);

  StreamedBufferReference<ByteAddressBuffer> material_buffer = MakeStreamedBufferReference(material_data, 0);

  float3 base_color = material_buffer.LoadFloat3();
  float3 subsurface_color = material_buffer.LoadFloat3();
  float subsurface = material_buffer.LoadFloat();
  float3 subsurface_radius = material_buffer.LoadFloat3();
  float metallic = material_buffer.LoadFloat();
  float specular = material_buffer.LoadFloat();
  float specular_tint = material_buffer.LoadFloat();

  float roughness = material_buffer.LoadFloat();
  float anisotropic = material_buffer.LoadFloat();
  float anisotropic_rotation = material_buffer.LoadFloat();
  float sheen = material_buffer.LoadFloat();
  float sheen_tint = material_buffer.LoadFloat();
  float clearcoat = material_buffer.LoadFloat();
  float clearcoat_roughness = material_buffer.LoadFloat();
  float ior = material_buffer.LoadFloat();
  float transmission = material_buffer.LoadFloat();
  float transmission_roughness = material_buffer.LoadFloat();

  float3 emission = material_buffer.LoadFloat3();
  float strength = material_buffer.LoadFloat();

  output.radiance = float4(emission * strength, 0.0);
  output.albedo_roughness = float4(base_color, roughness);
  output.position_specular = float4(input.world_position, specular);
  output.normal_metallic = float4(N * 0.5 + 0.5, metallic);
  output.stencil = 0;
  return output;
}
