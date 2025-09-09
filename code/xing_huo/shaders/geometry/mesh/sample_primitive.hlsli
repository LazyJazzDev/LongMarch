#pragma once
#include "common.hlsli"

template <class BufferType>
GeometryPrimitiveSample MeshSamplePrimitive(BufferType geometry_data, float3x4 transform, uint primitive_id, float2 sample) {
  uint num_indices = geometry_data.Load(4);
  uint position_offset = geometry_data.Load(8);
  uint position_stride = geometry_data.Load(12);
  uint index_offset = geometry_data.Load(48);
  uint tex_coord_offset = geometry_data.Load(24);
  uint tex_coord_stride = geometry_data.Load(28);
  uint3 vid;
  vid = geometry_data.Load3(index_offset + primitive_id * 3 * 4);
  if (sample.x + sample.y > 1.0f) {
    // Handle case where sample is outside the triangle
    sample = float2(1.0f - sample.x, 1.0f - sample.y);
  }
  float3 barycentrics = float3(1.0f - sample.x - sample.y, sample.x, sample.y);
  float3 pos[3];
  pos[0] = mul(transform, float4(LoadFloat3(geometry_data, position_offset + position_stride * vid[0]), 1.0));
  pos[1] = mul(transform, float4(LoadFloat3(geometry_data, position_offset + position_stride * vid[1]), 1.0));
  pos[2] = mul(transform, float4(LoadFloat3(geometry_data, position_offset + position_stride * vid[2]), 1.0));

  GeometryPrimitiveSample sample_result;
  sample_result.position = pos[0] * barycentrics[0] + pos[1] * barycentrics[1] + pos[2] * barycentrics[2];
  sample_result.normal = normalize(cross(pos[1] - pos[0], pos[2] - pos[0]));
  if (tex_coord_offset != 0) {
    sample_result.tex_coord = LoadFloat2(geometry_data, tex_coord_offset + tex_coord_stride * vid[0]) * barycentrics[0] +
                              LoadFloat2(geometry_data, tex_coord_offset + tex_coord_stride * vid[1]) * barycentrics[1] +
                              LoadFloat2(geometry_data, tex_coord_offset + tex_coord_stride * vid[2]) * barycentrics[2];
  } else {
    sample_result.tex_coord = float2(0.0f, 0.0f);
  }
  sample_result.pdf = 1.0f / (length(cross(pos[1] - pos[0], pos[2] - pos[0])) * 0.5f);
  return sample_result;
}
