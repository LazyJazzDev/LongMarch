#include "native_contract.hlsli"
#pragma once
#include "software/layout.hlsli"
#ifndef SOFTWARE_EXTERNAL_BINDINGS
#include "bindings.hlsli"
#else
#define SP_BINDING_software_instances software_instances
#endif

struct SoftwareHit {
  float distance;
  float2 barycentric;
  uint instance;
  uint primitive;
};

// Keep the compute renderer's hit ABI and ordered transparent-shadow logic.
// Force-opaque here means report triangle surfaces; material transmission is
// evaluated by the shared shading code after each closest hit.
bool InlineIntersect(SP_CONTEXT SP_RAY ray, bool any_hit, out SoftwareHit hit) {
  hit = (SoftwareHit)0;
  RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> query;
  query.TraceRayInline(query_scene, any_hit ? RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH : RAY_FLAG_NONE, 0xff, ray);
  while (query.Proceed()) {
  }
  if (query.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
    return false;
  hit.distance = query.CommittedRayT();
  hit.barycentric = query.CommittedTriangleBarycentrics();
  hit.instance = query.CommittedInstanceID();
  hit.primitive = query.CommittedPrimitiveIndex();
  return true;
}
