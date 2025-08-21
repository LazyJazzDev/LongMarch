#include "common.hlsli"

template <class BufferType, class BufferType2>
float PrimitivePower(BufferType material_data, BufferType2 geometry_data, float3x4 transform, uint primitive_id) {
  float area = PrimitiveArea(geometry_data, transform, primitive_id);
  float3 emission = LoadFloat3(material_data, 12);
  return max(max(emission.x, emission.y), emission.z) * area * PI * 2.0; // Use max to get the maximum power
}

template <class BufferType>
float3 EvaluateDirectLighting(BufferType material_data, float3 position, GeometryPrimitiveSample primitive_sample) {
  float3 emission = LoadFloat3(material_data, 12);
  return emission;
}
