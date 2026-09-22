#pragma once
#include "software/layout.hlsli"
#include "parameters.hlsli"
ConstantBuffer<RealtimeParameters> parameters : register(b0, space0);
ByteAddressBuffer scene_buffers[] : register(t0, space1);
#define instances scene_buffers[0]
ByteAddressBuffer geometry_buffers[] : register(t0, space2);

float3 VertexPosition(SoftwareInstance instance, uint vertex_id) {
  ByteAddressBuffer geometry = geometry_buffers[NonUniformResourceIndex(instance.geometry)];
  uint index = geometry.Load(geometry.Load(48) + vertex_id * 4);
  float3 position = asfloat(geometry.Load3(geometry.Load(8) + index * geometry.Load(12)));
  return mul(instance.object_to_world, float4(position, 1));
}

void Surface(float4 visibility, out float3 position, out float3 normal) {
  SoftwareInstance instance = LoadSoftwareInstance(instances, uint(visibility.x) - 1);
  uint vertex_offset = uint(visibility.y) * 3;
  float3 a = VertexPosition(instance, vertex_offset);
  float3 b = VertexPosition(instance, vertex_offset + 1);
  float3 c = VertexPosition(instance, vertex_offset + 2);
  position = a * (1 - visibility.z - visibility.w) + b * visibility.z + c * visibility.w;
  normal = normalize(cross(b - a, c - a));
  if (dot(normal, parameters.camera_position.xyz - position) < 0)
    normal = -normal;
}
