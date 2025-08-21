#include "common.hlsli"

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

HitRecord GetHitRecord(RayPayload payload) {
  HitRecord hit_record;
  BufferReference<ByteAddressBuffer> geometry_buffer = MakeBufferReference(data_buffers[InstanceID()], 0);
  GeometryHeader header;
  header = geometry_buffer.Load<GeometryHeader>(0);

  uint3 vid;
  vid = geometry_buffer.Load3(header.index_offset + PrimitiveIndex() * 3 * 4);

  float3 pos[3] = {LoadFloat3(geometry_buffer, header.position_offset + header.position_stride * vid[0]),
                   LoadFloat3(geometry_buffer, header.position_offset + header.position_stride * vid[1]),
                   LoadFloat3(geometry_buffer, header.position_offset + header.position_stride * vid[2])};

  float3 barycentrics = payload.primitive_coord;

  hit_record.front_facing = true;
  hit_record.position = pos[0] * barycentrics[0] + pos[1] * barycentrics[1] + pos[2] * barycentrics[2];
  hit_record.position = mul(ObjectToWorld3x4(), float4(hit_record.position, 1.0));
  hit_record.geom_normal = normalize(mul(WorldToObject4x3(), cross(pos[1] - pos[0], pos[2] - pos[0])).xyz);
  hit_record.pdf = 1.0f / (length(cross(pos[1] - pos[0], pos[2] - pos[0])) * 0.5f);

  if (header.normal_offset != 0) {
    hit_record.normal =
        LoadFloat3(geometry_buffer, header.normal_offset + header.normal_stride * vid[0]) * barycentrics[0] +
        LoadFloat3(geometry_buffer, header.normal_offset + header.normal_stride * vid[1]) * barycentrics[1] +
        LoadFloat3(geometry_buffer, header.normal_offset + header.normal_stride * vid[2]) * barycentrics[2];
    hit_record.normal = normalize(mul(WorldToObject4x3(), hit_record.normal).xyz);
  } else {
    hit_record.normal = hit_record.geom_normal;
  }
  // normal transformation need to multiply inverse transpose of the object to world matrix
  if (header.tex_coord_offset != 0) {
    hit_record.tex_coord =
        LoadFloat2(geometry_buffer, header.tex_coord_offset + header.tex_coord_stride * vid[0]) * barycentrics[0] +
        LoadFloat2(geometry_buffer, header.tex_coord_offset + header.tex_coord_stride * vid[1]) * barycentrics[1] +
        LoadFloat2(geometry_buffer, header.tex_coord_offset + header.tex_coord_stride * vid[2]) * barycentrics[2];
  } else {
    hit_record.tex_coord = float2(0.0, 0.0);
  }

  if (header.tangent_offset != 0) {
    hit_record.tangent =
        LoadFloat3(geometry_buffer, header.tangent_offset + header.tangent_stride * vid[0]) * barycentrics[0] +
        LoadFloat3(geometry_buffer, header.tangent_offset + header.tangent_stride * vid[1]) * barycentrics[1] +
        LoadFloat3(geometry_buffer, header.tangent_offset + header.tangent_stride * vid[2]) * barycentrics[2];
    hit_record.tangent = normalize(mul(ObjectToWorld3x4(), float4(hit_record.tangent, 0.0)).xyz);
  } else {
    hit_record.tangent = cross(hit_record.normal, float3(0.0, 0.0, 1.0));
    if (length(hit_record.tangent) < 0.001) {
      hit_record.tangent = cross(hit_record.normal, float3(1.0, 0.0, 0.0));
    }
    hit_record.tangent = normalize(hit_record.tangent);
  }

  if (header.signal_offset != 0) {
    hit_record.signal =
        LoadFloat(geometry_buffer, header.signal_offset + header.signal_stride * vid[0]) * barycentrics[0] +
        LoadFloat(geometry_buffer, header.signal_offset + header.signal_stride * vid[1]) * barycentrics[1] +
        LoadFloat(geometry_buffer, header.signal_offset + header.signal_stride * vid[2]) * barycentrics[2];
  } else {
    hit_record.signal = 1.0;
  }

  if (dot(WorldRayDirection(), hit_record.normal) > 0.0) {
    hit_record.front_facing = false;
    hit_record.geom_normal = -hit_record.geom_normal;
    hit_record.normal = -hit_record.normal;
    hit_record.tangent = -hit_record.tangent;
    hit_record.signal = -hit_record.signal;
  }

  hit_record.object_index = InstanceIndex();
  hit_record.primitive_index = PrimitiveIndex();
  return hit_record;
}
