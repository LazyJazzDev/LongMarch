#pragma once
#include "bindings.hlsli"

float ShadowRayNoAlpha(float3 origin, float3 direction, float dist) {
  RayDesc ray;
  ray.Origin = origin;
  ray.Direction = direction;
  ray.TMin = T_MIN * max(length(origin), 1.0);
  ray.TMax = dist;
#if defined(__spirv__) && defined(DEBUG_SHADER)
  RayPayload payload;
  TraceRay(as, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, payload);
  if (payload.t != -1.0) {
    return 0.0f;
  }
#else
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
#endif
  return 1.0;
}


float ShadowRay(float3 origin, float3 direction, float dist) {
  return 1.0;
}
