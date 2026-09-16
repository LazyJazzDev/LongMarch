#include "bindings.hlsli"
#include "direct_lighting.hlsli"
#include "geometry/mesh/hit_record.hlsli"
#include "material_sampler.hlsli"
#include "random.hlsli"

HitRecord MakeHitRecord(in BuiltInTriangleIntersectionAttributes attr) {
  return MakeMeshHitRecord(InstanceID(), InstanceIndex(), PrimitiveIndex(), attr.barycentrics,
                           RayTCurrent(), WorldRayDirection(), ObjectToWorld3x4(), WorldToObject4x3());
}
[shader("closesthit")] void RenderClosestHit(inout RenderContext context,
                                             in BuiltInTriangleIntersectionAttributes attr) {
  SampleMaterial(context, MakeHitRecord(attr));
}
[shader("closesthit")] void ShadowClosestHit(inout ShadowRayPayload payload,
                                             in BuiltInTriangleIntersectionAttributes attr) {
#ifdef SAMPLE_SHADOW_NO_HITRECORD
  SampleShadow(payload);
#else
  SampleShadow(payload, MakeHitRecord(attr));
#endif
}
