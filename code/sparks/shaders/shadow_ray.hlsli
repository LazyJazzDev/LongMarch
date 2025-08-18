#pragma once
#include "bindings.hlsli"

float ShadowRayNoAlpha(float3 origin, float3 direction, float dist) {
  RayDesc ray;
  ray.Origin = origin;
  ray.Direction = direction;
  ray.TMin = T_MIN * max(length(origin), 1.0);
  ray.TMax = dist;
  RayQuery<RAY_FLAG_NONE> rq;
  rq.TraceRayInline(
      as,
      RAY_FLAG_NONE,
      0xFF,            // Instance mask (all)
      ray
  );
  while (rq.Proceed()) {
    if (rq.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) {
      rq.Abort();
      return 0.0f;
    }
  }
  return 1.0;
}


float ShadowRay(float3 origin, float3 direction, float dist) {
  RayDesc ray;
  ray.Origin = origin;
  ray.Direction = direction;
  ray.TMin = T_MIN * max(length(origin), 1.0);
  ray.TMax = dist;
  ShadowRayPayload payload;
  payload.shadow = 1.0;
  TraceRay(as, RAY_FLAG_NONE, 0xFF, 1, 0, 1, ray, payload);
  return payload.shadow;
}
