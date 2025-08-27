#include "common.hlsli"

template <class BufferType>
float3 EvaluateDirectLighting(BufferType material_data, float3 position, GeometryPrimitiveSample primitive_sample) {
  return float3(0.0f, 0.0f, 0.0f); // Return zero contribution
}
