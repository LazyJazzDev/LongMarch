#pragma once
#include "software/layout.hlsli"

// The same hit record feeds the shared BSDFs, MIS and ordered transparent shadows.
struct SoftwareHit {
  float distance;
  float2 barycentric;
  uint instance;
  uint primitive;
};

[shader("closesthit")]
void LongMarchOptixClosest(inout SoftwareHit hit, BuiltInTriangleIntersectionAttributes attributes) {
  hit.distance = RayTCurrent();
  hit.barycentric = attributes.barycentrics;
  hit.instance = InstanceID();
  hit.primitive = PrimitiveIndex();
}

[shader("miss")]
void LongMarchOptixMiss(inout SoftwareHit hit) {
  hit.instance = SOFTWARE_INVALID;
}

bool InlineIntersect(RayDesc ray, bool any_hit, out SoftwareHit hit) {
  hit = (SoftwareHit)0;
  hit.instance = hit.primitive = SOFTWARE_INVALID;
  hit.distance = ray.TMax;
  // An empty IAS has handle zero and must not be passed to optixTrace.
  if (software_instances.Load(0) == 0)
    return false;
  uint flags = RAY_FLAG_FORCE_OPAQUE;
  if (any_hit) flags |= RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;
  TraceRay(query_scene, flags, 0xff, 0, 1, 0, ray, hit);
  return hit.instance != SOFTWARE_INVALID;
}
