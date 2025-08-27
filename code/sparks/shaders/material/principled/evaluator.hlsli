#include "common.hlsli"

template <class BufferType>
float3 EvaluateDirectLighting(BufferType material_data, float3 position, GeometryPrimitiveSample primitive_sample) {
  float4 emission = LoadFloat4(material_data, 92);
  return emission.xyz * emission.w; // Scale by the emission intensity
}
