#include "common.hlsli"

template <class BufferType>
float3 EvaluateDirectLighting(BufferType material_data, float3 position, GeometryPrimitiveSample primitive_sample) {
  float3 emission = LoadFloat3(material_data, 12);
  return emission;
}
