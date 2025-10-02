
struct SceneSettings {
  float3 ambient_light;
};

Texture2D<float4> albedo_roughness : register(t0, space0);
Texture2D<float4> position_specular : register(t0, space1);
Texture2D<float4> normal_metallic : register(t0, space2);
ConstantBuffer<SceneSettings> scene_settings : register(b0, space3);

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
