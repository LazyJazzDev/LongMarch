#include "../../native_contract.hlsli"

struct SceneSettings {
  float3 ambient_light;
};

struct CameraInfo {
  float4x4 view;
  float4x4 proj;
  float4x4 view_proj;
  float4x4 inv_view;
  float4x4 inv_proj;
  float4x4 inv_view_proj;
};

SP_RESOURCE(SP_TEXTURE(float4), albedo_roughness, t0, 0);
#define SP_BINDING_albedo_roughness SP_RESOURCE_ACCESS(SP_TEXTURE(float4), albedo_roughness, 0)
SP_RESOURCE(SP_TEXTURE(float4), position_specular, t0, 1);
#define SP_BINDING_position_specular SP_RESOURCE_ACCESS(SP_TEXTURE(float4), position_specular, 1)
SP_RESOURCE(SP_TEXTURE(float4), normal_metallic, t0, 2);
#define SP_BINDING_normal_metallic SP_RESOURCE_ACCESS(SP_TEXTURE(float4), normal_metallic, 2)
SP_RESOURCE(ConstantBuffer<CameraInfo>, camera_data, b0, 3);
#define SP_BINDING_camera_data SP_RESOURCE_ACCESS(ConstantBuffer<CameraInfo>, camera_data, 3)
SP_RESOURCE(ConstantBuffer<SceneSettings>, scene_settings, b0, 4);
#define SP_BINDING_scene_settings SP_RESOURCE_ACCESS(ConstantBuffer<SceneSettings>, scene_settings, 4)

struct VSOutput {
  float4 position : SV_POSITION;
};

VSOutput VSMain(uint vertex_id : SV_VertexID) {
  float2 pos[] = {float2(-1.0, -1.0), float2(1.0, -1.0), float2(-1.0, 1.0),
                  float2(-1.0, 1.0),  float2(1.0, -1.0), float2(1.0, 1.0)};
  VSOutput SP_BINDING_output;
  SP_BINDING_output.position = float4(pos[vertex_id], 0.0, 1.0);
  return SP_BINDING_output;
}

struct PSInput {
  float4 position : SV_POSITION;
};

struct PSOutput {
  float4 radiance : SV_TARGET0;
};

PSOutput PSMain(PSInput input) {
  PSOutput SP_BINDING_output;
  uint2 pixel_coords = uint2(input.position.xy);
  float3 albedo = SP_BINDING_albedo_roughness.Load(int3(pixel_coords, 0)).xyz;
  float3 position = SP_BINDING_position_specular.Load(int3(pixel_coords, 0)).xyz;
  float3 normal = SP_BINDING_normal_metallic.Load(int3(pixel_coords, 0)).xyz * 2.0 - 1.0;
  float3 N = normalize(normal);
  // float3 V = normalize(-position);
  // Ambient lighting
  float3 ambient_light = SP_BINDING_scene_settings.ambient_light;
  // Simple environment map (gradient)
  SP_BINDING_output.radiance = float4(albedo * ambient_light, 0.0);
  return SP_BINDING_output;
}
