#include "bindings.hlsli"
#include "direct_lighting.hlsli"
#include "geometry/mesh/geometry_header.hlsli"
#include "geometry/mesh/hit_record.hlsli"
#include "subsurface_random_walk.hlsli"

#include "material_sampler.hlsli"
#include "random.hlsli"

HitRecord MakeHitRecord(in BuiltInTriangleIntersectionAttributes attr) {
  return MakeMeshHitRecord(InstanceID(), InstanceIndex(), PrimitiveIndex(), attr.barycentrics, RayTCurrent(),
                           WorldRayDirection(), ObjectToWorld3x4(), WorldToObject4x3());
}

[shader("closesthit")] void RenderClosestHit(inout RenderContext context,
                                             in BuiltInTriangleIntersectionAttributes attr) {
  HitRecord hit_record = MakeHitRecord(attr);
  if (ContinueSubsurfaceRandomWalk(context, hit_record))
    return;
  SampleMaterial(context, hit_record);
}

    [shader("closesthit")] void ShadowClosestHit(inout ShadowRayPayload payload,
                                                 in BuiltInTriangleIntersectionAttributes attr) {
#if defined(SAMPLE_SHADOW_NO_HITRECORD)
  SampleShadow(payload);
#else
  SampleShadow(payload, MakeHitRecord(attr));
#endif
}

#if defined(SAMPLE_SHADOW_ANY_HIT)
[shader("anyhit")] void ShadowAnyHit(inout ShadowRayPayload payload, in BuiltInTriangleIntersectionAttributes attr) {
  float opacity = saturate(SampleShadowOpacity(MakeHitRecord(attr), WorldRayDirection()));
  payload.shadow *= 1.0f - opacity;
  if (payload.shadow > 1.0e-4f)
    IgnoreHit();
  else
    AcceptHitAndEndSearch();
}
#endif
