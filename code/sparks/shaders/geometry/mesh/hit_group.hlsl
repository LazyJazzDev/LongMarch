#include "bindings.hlsli"
#include "direct_lighting.hlsli"
#include "geometry/mesh/hit_record.hlsli"
#include "material_sampler.hlsli"
#include "random.hlsli"

[shader("closesthit")] void RenderClosestHit(inout RenderContext context,
                                             in BuiltInTriangleIntersectionAttributes attr) {
  RayPayload payload;
  payload.t = RayTCurrent();
  payload.primitive_coord =
      float3(1.0 - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x, attr.barycentrics.y);
  payload.transform = ObjectToWorld3x4();
  payload.instance_index = InstanceIndex();
  payload.instance_id = InstanceID();
  payload.primitive_index = PrimitiveIndex();

  HitRecord hit_record = GetHitRecord(payload);
  SampleMaterial(context, hit_record);
}

    [shader("closesthit")] void ShadowClosestHit(inout ShadowRayPayload payload,
                                                 in BuiltInTriangleIntersectionAttributes attr) {
#if defined(SAMPLE_SHADOW_NO_HITRECORD)
  SampleShadow(payload);
#else
  RayPayload payload;
  payload.t = RayTCurrent();
  payload.primitive_coord =
      float3(1.0 - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x, attr.barycentrics.y);
  payload.transform = ObjectToWorld3x4();
  payload.instance_index = InstanceIndex();
  payload.instance_id = InstanceID();
  payload.primitive_index = PrimitiveIndex();

  HitRecord hit_record = GetHitRecord(payload);

  SampleShadow(payload, hit_record);
#endif
}
