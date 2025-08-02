#pragma once
#include "bindings.hlsli"

float ShadowRay(float3 origin, float3 direction, float dist) {
  RayDesc ray;
  ray.Origin = origin;
  ray.Direction = direction;
  ray.TMin = T_MIN * length(origin);
  ray.TMax = dist;
  ShadowRayPayload payload;
  payload.shadow = 1.0;
  TraceRay(as, RAY_FLAG_NONE, 0xFF, 1, 0, 1, ray, payload);
  return payload.shadow;
}
