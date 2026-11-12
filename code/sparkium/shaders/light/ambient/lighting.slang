
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

RHI_TEXTURE2D(0, 0, float4, albedo_roughness);
RHI_TEXTURE2D(0, 1, float4, position_specular);
RHI_TEXTURE2D(0, 2, float4, normal_metallic);
RHI_UNIFORM(0, 3, CameraInfo, camera_data);
RHI_UNIFORM(0, 4, SceneSettings, scene_settings);

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
  float3 albedo = albedo_roughness.Load(int3(pixel_coords, 0)).xyz;
  float3 position = position_specular.Load(int3(pixel_coords, 0)).xyz;
  float3 normal = normal_metallic.Load(int3(pixel_coords, 0)).xyz * 2.0 - 1.0;
  float3 N = normalize(normal);
  // float3 V = normalize(-position);
  // Ambient lighting
  float3 ambient_light = scene_settings.ambient_light;
  // Simple environment map (gradient)
  output.radiance = float4(albedo * ambient_light, 0.0);
  return output;
}
