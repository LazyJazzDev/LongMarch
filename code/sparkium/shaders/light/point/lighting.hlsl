
#include "bsdf/principled_material.hlsli"
#include "common.hlsli"

struct PointLightSettings {
  float3 emission;
  float3 light_position;
};

struct CameraInfo {
  float4x4 view;
  float4x4 proj;
  float4x4 view_proj;
  float4x4 inv_view;
  float4x4 inv_proj;
  float4x4 inv_view_proj;
};

Texture2D<float4> albedo_roughness_buffer : register(t0, space0);
Texture2D<float4> position_specular_buffer : register(t0, space1);
Texture2D<float4> normal_metallic_buffer : register(t0, space2);
ConstantBuffer<CameraInfo> camera_data : register(b0, space3);
ByteAddressBuffer settings : register(t0, space4);

struct VSOutput {
  float4 position : SV_POSITION;
};

VSOutput VSMain(uint vertex_id : SV_VertexID) {
  float2 pos[] = {float2(-1.0, -1.0), float2(1.0, -1.0), float2(-1.0, 1.0),
                  float2(-1.0, 1.0),  float2(1.0, -1.0), float2(1.0, 1.0)};
  VSOutput output;
  output.position = float4(pos[vertex_id], 0.0, 1.0);
  return output;
}

struct PSInput {
  float4 position : SV_POSITION;
};

struct PSOutput {
  float4 radiance : SV_TARGET0;
};

PSOutput PSMain(PSInput input) {
  PSOutput output;
  uint2 pixel_coords = uint2(input.position.xy);
  float4 albedo_roughness = albedo_roughness_buffer.Load(int3(pixel_coords, 0));
  float4 position_specular = position_specular_buffer.Load(int3(pixel_coords, 0));
  float4 normal_metallic = normal_metallic_buffer.Load(int3(pixel_coords, 0));
  float3 albedo = albedo_roughness.xyz;
  float roughness = albedo_roughness.w;
  float3 position = position_specular.xyz;
  float specular = position_specular.w;
  float3 normal = normal_metallic.xyz * 2.0 - 1.0;
  float metallic = normal_metallic.w;
  float3 N = normalize(normal);
  float3 L = settings.Load<PointLightSettings>(0).light_position - position;
  float3 camera_position = transpose(camera_data.inv_view)[3].xyz;
  float3 V = normalize(camera_position - position);

  PrincipledMaterial material;
  material.hit_record.t = 0.0;
  material.hit_record.position = position;
  material.hit_record.tex_coord = float2(0.0, 0.0);
  material.hit_record.normal = N;
  material.hit_record.geom_normal = N;
  material.hit_record.tangent = float3(0.0, 0.0, 0.0);
  material.hit_record.signal = 1.0;
  material.hit_record.pdf = 1.0;
  material.hit_record.primitive_index = 0;
  material.hit_record.object_index = 0;
  material.hit_record.front_facing = true;

  material.omega_v = V;
  material.base_color = albedo;
  material.subsurface_color = float3(0, 0, 0);
  material.subsurface = 0.0;
  material.subsurface_radius = float3(1.0, 1.0, 1.0);
  material.metallic = metallic;
  material.specular = specular;
  material.specular_tint = 0.0;

  material.roughness = max(roughness, 0.08);
  material.anisotropic = 0.0;
  material.anisotropic_rotation = 0.0;
  material.sheen = 0.0;
  material.sheen_tint = 0.0;
  material.clearcoat = 0.0;
  material.clearcoat_roughness = 0.0;
  material.ior = 1.0;
  material.transmission = 0.0;
  material.transmission_roughness = 0.0;

  float pdf;

  // Ambient lighting
  float3 strength = settings.Load<PointLightSettings>(0).emission / (dot(L, L) * 4.0 * PI);
  L = normalize(L);
  albedo = material.EvalPrincipledBSDF(L, pdf);
  // strength *= max(dot(N, L), 0.0);
  // Simple environment map (gradient)
  output.radiance = float4(albedo * strength, 0.0);
  return output;
}
