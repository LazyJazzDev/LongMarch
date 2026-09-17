#pragma once
// Portable port of code/sparkium/shaders/camera.hlsl (CameraPinhole).

#include "sparkium/backends/core/rng.h"
#include "sparkium/backends/core/structs.h"

namespace sparkium::backends {

SPARKIUM_HD inline void CameraPinhole(const DeviceScene &scene, RayGenPayload &raygen_payload) {
  const float4x4 &camera_to_world = scene.camera_to_world;
  float2 scale = scene.camera_scale;
  float aperture_radius = scene.aperture_radius;
  float focus_distance = scene.focus_distance;
  int32_t aperture_blades = scene.aperture_blades;
  float aperture_rotation = scene.aperture_rotation;
  float aperture_ratio = scene.aperture_ratio;

  raygen_payload.origin = float3(0, 0, 0);
  raygen_payload.direction = device::normalize(float3(raygen_payload.uv * scale, -1.0f));
  if (aperture_radius > 0.0f && focus_distance > 0.0f) {
    float2 aperture_sample;
    if (aperture_blades >= 3) {
      float sector_position = raygen_payload.lens_sample.x * static_cast<float>(aperture_blades);
      int32_t sector = static_cast<int32_t>(sector_position);
      if (sector > aperture_blades - 1)
        sector = aperture_blades - 1;
      float edge_factor = device::frac(sector_position);
      float angle0 = aperture_rotation + 2.0f * SPARKIUM_PI * static_cast<float>(sector) /
                                              static_cast<float>(aperture_blades);
      float angle1 = aperture_rotation + 2.0f * SPARKIUM_PI * static_cast<float>(sector + 1) /
                                              static_cast<float>(aperture_blades);
      float2 vertex0 = float2(cosf(angle0), sinf(angle0));
      float2 vertex1 = float2(cosf(angle1), sinf(angle1));
      aperture_sample = sqrtf(raygen_payload.lens_sample.y) * device::lerp(vertex0, vertex1, edge_factor);
    } else {
      float radius = sqrtf(raygen_payload.lens_sample.x);
      float angle = 2.0f * SPARKIUM_PI * raygen_payload.lens_sample.y;
      aperture_sample = radius * float2(cosf(angle), sinf(angle));
    }
    aperture_sample.x *= device::max(aperture_ratio, 1e-6f);
    float3 lens_position = float3(aperture_radius * aperture_sample, 0.0f);
    float3 focus_position = raygen_payload.direction * (focus_distance / -raygen_payload.direction.z);
    raygen_payload.origin = lens_position;
    raygen_payload.direction = device::normalize(focus_position - lens_position);
  }
  raygen_payload.origin = mul(camera_to_world, float4(raygen_payload.origin, 1.0f)).xyz();
  raygen_payload.direction = device::normalize(mul(camera_to_world, float4(raygen_payload.direction, 0.0f)).xyz());
}

}  // namespace sparkium::backends
