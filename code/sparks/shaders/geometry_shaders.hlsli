#pragma once
// Geometry Sampler Implementation

template <class BufferType>
float PrimitiveArea(int shader_index, BufferType geometry_data, float3x4 transform, uint primitive_id) {
  switch (shader_index) {
// PrimitiveArea Function List
  default:
    return 0.0f;
  }
}

template <class BufferType>
GeometryPrimitiveSample SamplePrimitive(int shader_index, BufferType geometry_data, float3x4 transform, uint primitive_id, float2 sample) {
  switch (shader_index) {
// SamplePrimitive Function List
  default:
    break;
  }
  GeometryPrimitiveSample empty_sample;
  empty_sample.position = float3(0.0f, 0.0f, 0.0f);
  empty_sample.normal = float3(0.0f, 0.0f, 1.0f);
  empty_sample.tex_coord = float2(0.0f, 0.0f);
  empty_sample.pdf = 1.0f;
  return empty_sample;
}
