#pragma once
#include "common.hlsli"

// Hit Record Implementation

HitRecord GetHitRecord(int geometry_shader_index, RayPayload payload) {
  switch (geometry_shader_index) {
// GetHitRecord Function List
  default:
    break;
  }
  HitRecord hit_record;
  hit_record.t = payload.t;
  hit_record.position = float3(0.0f, 0.0f, 0.0f);
  hit_record.normal = float3(0.0f, 0.0f, 1.0f);
  hit_record.geom_normal = float3(0.0f, 0.0f, 1.0f);
  hit_record.tex_coord = float2(0.0f, 0.0f);
  hit_record.tangent = float3(0.0f, 0.0f, 1.0f);
  hit_record.signal = 0.0f;
  hit_record.pdf = 1.0f;
  hit_record.primitive_index = payload.primitive_index;
  hit_record.object_index = payload.instance_index;
  hit_record.front_facing = true;
  return hit_record;
}
