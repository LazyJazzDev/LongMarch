#pragma once
#include "common.slang"
#include "material/principled/eval_direct_light.slang"

__generic<BufferType : IBufferReader> struct MaterialEvaluator {
  BufferType material_data;

  __generic<GeometrySamplerType : IGeometrySampler> float PrimitivePower(GeometrySamplerType geometry_sampler,
                                                                         uint primitive_id) {
    float area = geometry_sampler.PrimitiveArea(primitive_id);
    float4 emission = LoadFloat4(material_data, 92);
    return max(max(emission.x, emission.y), emission.z) * emission.w * area * PI *
           2.0;  // Use max to get the maximum power
  }

  float3 EvaluateDirectLighting(float3 position, GeometryPrimitiveSample primitive_sample) {
    return MaterialPrincipledEvaluateDirectLighting(material_data, position, primitive_sample);
  }
};
