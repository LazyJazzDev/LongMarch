#include "bindings.hlsli"
#include "common.hlsli"
#include "random.hlsli"

struct GeometryHeader {
  uint num_vertices;
  uint num_indices;
  uint position_offset;
  uint position_stride;
  uint normal_offset;
  uint normal_stride;
  uint tex_coord_offset;
  uint tex_coord_stride;
  uint tangent_offset;
  uint tangent_stride;
  uint signal_offset;
  uint signal_stride;
  uint index_offset;
};

[shader("closesthit")] void ClosestHitMain(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr) {
  ByteAddressBuffer geometry_buffer = geometry_data[InstanceID()];
  GeometryHeader header;
  header.num_vertices = geometry_buffer.Load(0);
  header.num_indices = geometry_buffer.Load(4);
  header.position_offset = geometry_buffer.Load(8);
  header.position_stride = geometry_buffer.Load(12);
  header.normal_offset = geometry_buffer.Load(16);
  header.normal_stride = geometry_buffer.Load(20);
  header.tex_coord_offset = geometry_buffer.Load(24);
  header.tex_coord_stride = geometry_buffer.Load(28);
  header.tangent_offset = geometry_buffer.Load(32);
  header.tangent_stride = geometry_buffer.Load(36);
  header.signal_offset = geometry_buffer.Load(40);
  header.signal_stride = geometry_buffer.Load(44);
  header.index_offset = geometry_buffer.Load(48);

  uint vid[3];
  vid[0] = geometry_buffer.Load(header.index_offset + PrimitiveIndex() * 3 * 4);
  vid[1] = geometry_buffer.Load(header.index_offset + PrimitiveIndex() * 3 * 4 + 4);
  vid[2] = geometry_buffer.Load(header.index_offset + PrimitiveIndex() * 3 * 4 + 8);

  float3 pos[3] = {LoadFloat3(geometry_buffer, header.position_offset + header.position_stride * vid[0]),
                   LoadFloat3(geometry_buffer, header.position_offset + header.position_stride * vid[1]),
                   LoadFloat3(geometry_buffer, header.position_offset + header.position_stride * vid[2])};

  float3 barycentrics =
      float3(1.0 - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x, attr.barycentrics.y);

  payload.position = pos[0] * barycentrics[0] + pos[1] * barycentrics[1] + pos[2] * barycentrics[2];
  payload.position = mul(ObjectToWorld3x4(), float4(payload.position, 1.0));
  if (header.normal_offset != 0) {
    payload.normal =
        LoadFloat3(geometry_buffer, header.normal_offset + header.normal_stride * vid[0]) * barycentrics[0] +
        LoadFloat3(geometry_buffer, header.normal_offset + header.normal_stride * vid[1]) * barycentrics[1] +
        LoadFloat3(geometry_buffer, header.normal_offset + header.normal_stride * vid[2]) * barycentrics[2];
  } else {
    payload.normal = cross(pos[1] - pos[0], pos[2] - pos[0]);
  }
  // normal transformation need to multiply inverse transpose of the object to world matrix
  payload.normal = normalize(mul(WorldToObject4x3(), payload.normal).xyz);
  if (header.tex_coord_offset != 0) {
    payload.tex_coord =
        LoadFloat2(geometry_buffer, header.tex_coord_offset + header.tex_coord_stride * vid[0]) * barycentrics[0] +
        LoadFloat2(geometry_buffer, header.tex_coord_offset + header.tex_coord_stride * vid[1]) * barycentrics[1] +
        LoadFloat2(geometry_buffer, header.tex_coord_offset + header.tex_coord_stride * vid[2]) * barycentrics[2];
  } else {
    payload.tex_coord = float2(0.0, 0.0);
  }

  if (header.tangent_offset != 0) {
    payload.tangent =
        LoadFloat3(geometry_buffer, header.tangent_offset + header.tangent_stride * vid[0]) * barycentrics[0] +
        LoadFloat3(geometry_buffer, header.tangent_offset + header.tangent_stride * vid[1]) * barycentrics[1] +
        LoadFloat3(geometry_buffer, header.tangent_offset + header.tangent_stride * vid[2]) * barycentrics[2];
    payload.tangent = normalize(mul(ObjectToWorld3x4(), float4(payload.tangent, 0.0)).xyz);
  } else {
    payload.tangent = float3(0.0, 0.0, 0.0);
  }

  if (header.signal_offset != 0) {
    payload.signal =
        LoadFloat(geometry_buffer, header.signal_offset + header.signal_stride * vid[0]) * barycentrics[0] +
        LoadFloat(geometry_buffer, header.signal_offset + header.signal_stride * vid[1]) * barycentrics[1] +
        LoadFloat(geometry_buffer, header.signal_offset + header.signal_stride * vid[2]) * barycentrics[2];
  } else {
    payload.signal = 1.0;
  }

  if (dot(WorldRayDirection(), payload.normal) > 0.0) {
    payload.front_facing = false;
    payload.normal = -payload.normal;
    payload.tangent = -payload.tangent;
  }

  payload.object_id = InstanceIndex();
}
