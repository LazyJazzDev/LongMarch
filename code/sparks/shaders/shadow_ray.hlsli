#pragma once
#include "bindings.hlsli"

float ShadowRay(float3 origin, float3 direction, float dist) {
  RayDesc ray;
  ray.Origin = origin;
  ray.Direction = direction;
  ray.TMin = T_MIN * length(origin);
  ray.TMax = dist;
  HitRecord hit_record;
  TraceRay(as, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, hit_record);
  if (hit_record.t != -1.0) {
    return 0.0;
  }
  return 1.0;
}
