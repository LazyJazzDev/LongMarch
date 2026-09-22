#include "bindings.hlsli"
#include "common.hlsli"

void CameraPinhole(inout RayGenPayload raygen_payload) {
  float4x4 camera_to_world;
  float2 scale;
  camera_to_world = LoadFloat4x4(camera_data, 64);
  scale = LoadFloat2(camera_data, 128);
  float aperture_radius = LoadFloat(camera_data, 136);
  float focus_distance = LoadFloat(camera_data, 140);
  int aperture_blades = camera_data.Load<int>(144);
  float aperture_rotation = LoadFloat(camera_data, 148);
  float aperture_ratio = LoadFloat(camera_data, 152);

  raygen_payload.origin = float3(0, 0, 0);
  raygen_payload.direction = normalize(float3(raygen_payload.uv * scale, -1));
  if (aperture_radius > 0.0f && focus_distance > 0.0f) {
    float2 aperture_sample;
    if (aperture_blades >= 3) {
      float sector_position = raygen_payload.lens_sample.x * aperture_blades;
      int sector = min(int(sector_position), aperture_blades - 1);
      float edge_factor = frac(sector_position);
      float angle0 = aperture_rotation + 2.0f * PI * sector / aperture_blades;
      float angle1 = aperture_rotation + 2.0f * PI * (sector + 1) / aperture_blades;
      float2 vertex0 = float2(cos(angle0), sin(angle0));
      float2 vertex1 = float2(cos(angle1), sin(angle1));
      aperture_sample = sqrt(raygen_payload.lens_sample.y) * lerp(vertex0, vertex1, edge_factor);
    } else {
      float radius = sqrt(raygen_payload.lens_sample.x);
      float angle = 2.0f * PI * raygen_payload.lens_sample.y;
      aperture_sample = radius * float2(cos(angle), sin(angle));
    }
    aperture_sample.x *= max(aperture_ratio, 1e-6f);
    float3 lens_position = float3(aperture_radius * aperture_sample, 0.0f);
    float3 focus_position = raygen_payload.direction * (focus_distance / -raygen_payload.direction.z);
    raygen_payload.origin = lens_position;
    raygen_payload.direction = normalize(focus_position - lens_position);
  }
  raygen_payload.origin = mul(camera_to_world, float4(raygen_payload.origin, 1.0)).xyz;
  raygen_payload.direction = normalize(mul(camera_to_world, float4(raygen_payload.direction, 0.0)).xyz);
}
