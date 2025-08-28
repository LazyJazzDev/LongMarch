#pragma once
#include "common.hlsli"
#include "material/lambertian/eval_direct_light.hlsli"

template <class BufferType>
class MaterialEvaluator {
  BufferType material_data;

  template <class GeometrySamplerType>
  float PrimitivePower(GeometrySamplerType geometry_sampler, uint primitive_id) {
    float area = geometry_sampler.PrimitiveArea(primitive_id);
    float3 emission = LoadFloat3(material_data, 12);
    return max(max(emission.x, emission.y), emission.z) * area * PI * 2.0; // Use max to get the maximum power
  }

  float3 EvaluateDirectLighting(float3 position, GeometryPrimitiveSample primitive_sample) {
    return MaterialLambertianEvaluateDirectLighting(material_data, position, primitive_sample);
  }
};
