#include "compute_contract.hlsli"
#include "bindings.hlsli"
#include "direct_lighting.hlsli"
#include "geometry/mesh/geometry_header.hlsli"
#include "geometry/mesh/hit_record.hlsli"
#include "subsurface_random_walk.hlsli"

#include "material_sampler.hlsli"
#include "random.hlsli"

HitRecord MakeHitRecord(SP_CONTEXT in BuiltInTriangleIntersectionAttributes attr) {
  return MakeMeshHitRecord(SP_CONTEXT_ARG InstanceID(), InstanceIndex(), PrimitiveIndex(), attr.barycentrics,
                           RayTCurrent(), WorldRayDirection(), ObjectToWorld3x4(), WorldToObject4x3());
}

[shader("closesthit")] void RenderClosestHit(SP_CONTEXT inout RenderContext context,
                                             in BuiltInTriangleIntersectionAttributes attr) {
  HitRecord hit_record = MakeHitRecord(SP_CONTEXT_ARG attr);
  if (ContinueSubsurfaceRandomWalk(SP_CONTEXT_ARG context, hit_record))
    return;
  SampleMaterial(SP_CONTEXT_ARG context, hit_record);
}

    [shader("closesthit")] void ShadowClosestHit(SP_CONTEXT inout ShadowRayPayload payload,
                                                 in BuiltInTriangleIntersectionAttributes attr) {
#if defined(SAMPLE_SHADOW_NO_HITRECORD)
  SampleShadow(payload);
#else
  SampleShadow(payload, MakeHitRecord(SP_CONTEXT_ARG attr));
#endif
}

#if defined(SAMPLE_SHADOW_ANY_HIT)
[shader("anyhit")] void ShadowAnyHit(SP_CONTEXT inout ShadowRayPayload payload,
                                     in BuiltInTriangleIntersectionAttributes attr) {
  float opacity = saturate(SampleShadowOpacity(SP_CONTEXT_ARG MakeHitRecord(SP_CONTEXT_ARG attr), WorldRayDirection()));
  payload.shadow *= 1.0f - opacity;
  if (payload.shadow > 1.0e-4f)
    IgnoreHit();
  else
    AcceptHitAndEndSearch();
}
#endif
