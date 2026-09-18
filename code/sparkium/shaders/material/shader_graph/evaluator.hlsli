#ifndef SPARKIUM_SHADER_MATERIAL_SHADER_GRAPH_EVALUATOR_HLSLI_
#define SPARKIUM_SHADER_MATERIAL_SHADER_GRAPH_EVALUATOR_HLSLI_
#include "common.hlsli"

template <class BufferType>
class MaterialEvaluator {
  BufferType material_data;

  template <class GeometrySamplerType>
  float PrimitivePower(GeometrySamplerType geometry_sampler, uint primitive_id) {
    float3 emission = LoadFloat3(material_data, 0);
    return max(emission.x, max(emission.y, emission.z)) * geometry_sampler.PrimitiveArea(primitive_id) * PI * 2.0f;
  }

  float3 EvaluateDirectLighting(float3 position, GeometryPrimitiveSample primitive_sample) {
    return LoadFloat3(material_data, 0);
  }
};

#endif  // SPARKIUM_SHADER_MATERIAL_SHADER_GRAPH_EVALUATOR_HLSLI_