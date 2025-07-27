#include "bindings.hlsli"
#include "common.hlsli"

[shader("callable")] void CameraPinhole(inout RayGenPayload raygen_payload) {
  float4x4 camera_to_world;
  float2 scale;
  camera_to_world = LoadFloat4x4(camera_data, 64);
  scale = LoadFloat2(camera_data, 128);

  raygen_payload.origin = float3(0, 0, 0);
  raygen_payload.direction = normalize(float3(raygen_payload.uv * scale, -1));
  raygen_payload.origin = mul(camera_to_world, float4(raygen_payload.origin, 1.0)).xyz;
  raygen_payload.direction = normalize(mul(camera_to_world, float4(raygen_payload.direction, 0.0)).xyz);
}
