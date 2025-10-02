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

struct MaterialLambertian {
  float3 base_color;
  float3 emission;
};

StructuredBuffer<MaterialLambertian> material_data : register(t0, space2);

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
  float3 albedo = material_data[0].base_color;
  float3 emission = material_data[0].emission;
  output.radiance = float4(emission, 0.0);
  output.albedo_roughness = float4(albedo, 1.0);
  output.position_specular = float4(input.world_position, 0.0);
  output.normal_metallic = float4(N * 0.5 + 0.5, 0.0);
  output.stencil = 0;
  return output;
}
