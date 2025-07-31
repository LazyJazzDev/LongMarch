#include "bindings.hlsli"

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

[shader("closesthit")] void ClosestHitMain(inout HitRecord hit_group, in BuiltInTriangleIntersectionAttributes attr) {
  BufferReference<ByteAddressBuffer> geometry_buffer = MakeBufferReference(data_buffers[InstanceID()], 0);
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

  uint3 vid;
  vid = geometry_buffer.Load3(header.index_offset + PrimitiveIndex() * 3 * 4);

  float3 pos[3] = {LoadFloat3(geometry_buffer, header.position_offset + header.position_stride * vid[0]),
                   LoadFloat3(geometry_buffer, header.position_offset + header.position_stride * vid[1]),
                   LoadFloat3(geometry_buffer, header.position_offset + header.position_stride * vid[2])};

  float3 barycentrics =
      float3(1.0 - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x, attr.barycentrics.y);

  hit_group.front_facing = true;
  hit_group.position = pos[0] * barycentrics[0] + pos[1] * barycentrics[1] + pos[2] * barycentrics[2];
  hit_group.position = mul(ObjectToWorld3x4(), float4(hit_group.position, 1.0));
  hit_group.geom_normal = normalize(mul(WorldToObject4x3(), cross(pos[1] - pos[0], pos[2] - pos[0])).xyz);

  if (header.normal_offset != 0) {
    hit_group.normal =
        LoadFloat3(geometry_buffer, header.normal_offset + header.normal_stride * vid[0]) * barycentrics[0] +
        LoadFloat3(geometry_buffer, header.normal_offset + header.normal_stride * vid[1]) * barycentrics[1] +
        LoadFloat3(geometry_buffer, header.normal_offset + header.normal_stride * vid[2]) * barycentrics[2];
    hit_group.normal = normalize(mul(WorldToObject4x3(), hit_group.normal).xyz);
  } else {
    hit_group.normal = hit_group.geom_normal;
  }
  // normal transformation need to multiply inverse transpose of the object to world matrix
  if (header.tex_coord_offset != 0) {
    hit_group.tex_coord =
        LoadFloat2(geometry_buffer, header.tex_coord_offset + header.tex_coord_stride * vid[0]) * barycentrics[0] +
        LoadFloat2(geometry_buffer, header.tex_coord_offset + header.tex_coord_stride * vid[1]) * barycentrics[1] +
        LoadFloat2(geometry_buffer, header.tex_coord_offset + header.tex_coord_stride * vid[2]) * barycentrics[2];
  } else {
    hit_group.tex_coord = float2(0.0, 0.0);
  }

  if (header.tangent_offset != 0) {
    hit_group.tangent =
        LoadFloat3(geometry_buffer, header.tangent_offset + header.tangent_stride * vid[0]) * barycentrics[0] +
        LoadFloat3(geometry_buffer, header.tangent_offset + header.tangent_stride * vid[1]) * barycentrics[1] +
        LoadFloat3(geometry_buffer, header.tangent_offset + header.tangent_stride * vid[2]) * barycentrics[2];
    hit_group.tangent = normalize(mul(ObjectToWorld3x4(), float4(hit_group.tangent, 0.0)).xyz);
  } else {
    hit_group.tangent = float3(0.0, 0.0, 0.0);
  }

  if (header.signal_offset != 0) {
    hit_group.signal =
        LoadFloat(geometry_buffer, header.signal_offset + header.signal_stride * vid[0]) * barycentrics[0] +
        LoadFloat(geometry_buffer, header.signal_offset + header.signal_stride * vid[1]) * barycentrics[1] +
        LoadFloat(geometry_buffer, header.signal_offset + header.signal_stride * vid[2]) * barycentrics[2];
  } else {
    hit_group.signal = 1.0;
  }

  if (dot(WorldRayDirection(), hit_group.normal) > 0.0) {
    hit_group.front_facing = false;
    hit_group.geom_normal = -hit_group.geom_normal;
    hit_group.normal = -hit_group.normal;
    hit_group.tangent = -hit_group.tangent;
    hit_group.signal = -hit_group.signal;
  }

  hit_group.object_id = InstanceIndex();
  hit_group.primitive_id = PrimitiveIndex();
}
