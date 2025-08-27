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

  HitRecord hit_record = GetHitRecord(payload, WorldRayDirection());
  SampleMaterial(context, hit_record);
}
