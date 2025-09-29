#pragma once
#include "geometry/mesh/sample_primitive.hlsli"

template <class BufferType>
class GeometrySampler {
  BufferType geometry_data;
  float3x4 transform;

  void SetTransform(float3x4 new_transform) {
    transform = new_transform;
  }

  float PrimitiveArea(uint primitive_id) {
    uint num_indices = geometry_data.Load(4);
    uint position_offset = geometry_data.Load(8);
    uint position_stride = geometry_data.Load(12);
    uint index_offset = geometry_data.Load(48);
    uint3 vid;
    vid = geometry_data.Load3(index_offset + primitive_id * 3 * 4);
    float3 pos[3];
    pos[0] = mul(transform, float4(LoadFloat3(geometry_data, position_offset + position_stride * vid[0]), 1.0));
    pos[1] = mul(transform, float4(LoadFloat3(geometry_data, position_offset + position_stride * vid[1]), 1.0));
    pos[2] = mul(transform, float4(LoadFloat3(geometry_data, position_offset + position_stride * vid[2]), 1.0));
    return length(cross(pos[1] - pos[0], pos[2] - pos[0])) * 0.5f;
  }

  GeometryPrimitiveSample SamplePrimitive(uint primitive_id, float2 sample) {
    return MeshSamplePrimitive(geometry_data, transform, primitive_id, sample);
  }

};
