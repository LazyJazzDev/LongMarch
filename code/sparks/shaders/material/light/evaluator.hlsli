#pragma once
#include "material/light/eval_direct_light.hlsli"
template <class BufferType>
class MaterialEvaluator {
  BufferType material_data;

  template <class GeometrySamplerType>
  float PrimitivePower(GeometrySamplerType geometry_sampler, uint primitive_id) {
    float area = geometry_sampler.PrimitiveArea(primitive_id);
    float3 emission = LoadFloat3(material_data, 0); // Assuming power is stored at offset 4
    uint two_sided = material_data.Load(12); // Assuming two_sided is stored at offset 12
    float result = max(max(emission.x, emission.y), emission.z) * area * PI; // Use max to get the maximum power
    if (two_sided) {
      result *= 2.0f; // If two-sided, double the power
    }
    return result;
  }

  float3 EvaluateDirectLighting(float3 position, GeometryPrimitiveSample primitive_sample) {
    return MaterialLightEvaluateDirectLighting(material_data, position, primitive_sample);
  }
};
