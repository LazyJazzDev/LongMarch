#include "common.hlsli"

template <class BufferType>
class MaterialEvaluator {
  BufferType material_data;

  template <class GeometrySamplerType>
  float PrimitivePower(GeometrySamplerType geometry_sampler, uint primitive_id) {
    float area = geometry_sampler.PrimitiveArea(primitive_id);
    float4 emission = LoadFloat4(material_data, 92);
    return max(max(emission.x, emission.y), emission.z) * emission.w * area * PI * 2.0; // Use max to get the maximum power
  }

  float3 EvaluateDirectLighting(float3 position, GeometryPrimitiveSample primitive_sample) {
    float4 emission = LoadFloat4(material_data, 92);
    return emission.xyz * emission.w; // Scale by the emission intensity
  }
};
