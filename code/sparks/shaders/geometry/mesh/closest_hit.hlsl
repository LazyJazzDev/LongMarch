#include "bindings.hlsli"
#include "common.hlsli"

[shader("closesthit")] void ClosestHit(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr) {
  payload.t = RayTCurrent();
  payload.primitive_coord =
      float3(1.0 - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x, attr.barycentrics.y);
  payload.transform = ObjectToWorld3x4();
  payload.instance_index = InstanceIndex();
  payload.instance_id = InstanceID();
  payload.primitive_index = PrimitiveIndex();
}
