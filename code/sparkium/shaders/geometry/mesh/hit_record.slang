#pragma once
#include "bindings.slang"
#include "geometry/mesh/geometry_header.slang"

HitRecord MakeMeshHitRecord(uint geometry_index,
                            uint object_index,
                            uint primitive_index,
                            float2 barycentric,
                            float distance,
                            float3 ray_direction,
                            float3x4 object_to_world,
                            float4x3 world_to_object) {
  HitRecord hit_record;
  BufferReference<ByteAddressBuffer> geometry_buffer =
      MakeBufferReference(data_buffers[NonUniformResourceIndex(geometry_index)], 0);
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
  header.color_offset = geometry_buffer.Load(52);
  header.color_stride = geometry_buffer.Load(56);

  uint3 vid;
  vid = geometry_buffer.Load3(header.index_offset + primitive_index * 3 * 4);

  float3 pos[3] = {LoadFloat3(geometry_buffer, header.position_offset + header.position_stride * vid[0]),
                   LoadFloat3(geometry_buffer, header.position_offset + header.position_stride * vid[1]),
                   LoadFloat3(geometry_buffer, header.position_offset + header.position_stride * vid[2])};

  float3 barycentrics = float3(1.0 - barycentric.x - barycentric.y, barycentric.x, barycentric.y);

  hit_record.t = distance;
  hit_record.front_facing = true;
  hit_record.position = pos[0] * barycentrics[0] + pos[1] * barycentrics[1] + pos[2] * barycentrics[2];
  hit_record.object_position = hit_record.position;
  hit_record.object_origin = mul(object_to_world, float4(0, 0, 0, 1));
  float3 world_pos[3] = {mul(object_to_world, float4(pos[0], 1.0)).xyz, mul(object_to_world, float4(pos[1], 1.0)).xyz,
                         mul(object_to_world, float4(pos[2], 1.0)).xyz};
  hit_record.position =
      world_pos[0] * barycentrics[0] + world_pos[1] * barycentrics[1] + world_pos[2] * barycentrics[2];
  hit_record.geom_normal = normalize(mul(world_to_object, cross(pos[1] - pos[0], pos[2] - pos[0])).xyz);
  // Light-hit MIS is evaluated in world-space solid angle. Using the local
  // triangle area here loses most of the BSDF-sampled contribution for scaled
  // emitters (for example a unit quad scaled to a 30 x 30 area light).
  hit_record.pdf = 1.0f / (length(cross(world_pos[1] - world_pos[0], world_pos[2] - world_pos[0])) * 0.5f);

  if (header.normal_offset != 0) {
    hit_record.normal =
        LoadFloat3(geometry_buffer, header.normal_offset + header.normal_stride * vid[0]) * barycentrics[0] +
        LoadFloat3(geometry_buffer, header.normal_offset + header.normal_stride * vid[1]) * barycentrics[1] +
        LoadFloat3(geometry_buffer, header.normal_offset + header.normal_stride * vid[2]) * barycentrics[2];
    hit_record.normal = normalize(mul(world_to_object, hit_record.normal).xyz);
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
  if (header.color_offset != 0) {
    hit_record.color =
        LoadFloat3(geometry_buffer, header.color_offset + header.color_stride * vid[0]) * barycentrics[0] +
        LoadFloat3(geometry_buffer, header.color_offset + header.color_stride * vid[1]) * barycentrics[1] +
        LoadFloat3(geometry_buffer, header.color_offset + header.color_stride * vid[2]) * barycentrics[2];
  } else {
    hit_record.color = float3(1.0, 1.0, 1.0);
  }

  if (header.tangent_offset != 0) {
    hit_record.tangent =
        LoadFloat3(geometry_buffer, header.tangent_offset + header.tangent_stride * vid[0]) * barycentrics[0] +
        LoadFloat3(geometry_buffer, header.tangent_offset + header.tangent_stride * vid[1]) * barycentrics[1] +
        LoadFloat3(geometry_buffer, header.tangent_offset + header.tangent_stride * vid[2]) * barycentrics[2];
    hit_record.tangent = normalize(mul(object_to_world, float4(hit_record.tangent, 0.0)).xyz);
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
    // Zero marks the absence of a valid tangent frame. Tangent-space normal
    // maps must then leave the interpolated surface normal unchanged.
    hit_record.signal = 0.0;
  }

  if (dot(ray_direction, hit_record.normal) > 0.0) {
    hit_record.front_facing = false;
    hit_record.geom_normal = -hit_record.geom_normal;
    hit_record.normal = -hit_record.normal;
    hit_record.tangent = -hit_record.tangent;
    hit_record.signal = -hit_record.signal;
  }

  hit_record.object_index = object_index;
  hit_record.primitive_index = primitive_index;

  return hit_record;
}
