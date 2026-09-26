#pragma once
#include "common.slang"

__generic<BufferType : IBufferReader> struct MaterialEvaluator {
  BufferType material_data;

  __generic<GeometrySamplerType : IGeometrySampler> float PrimitivePower(GeometrySamplerType geometry_sampler,
                                                                         uint primitive_id) {
    return 0.0;  // Use max to get the maximum power
  }

  float3 EvaluateDirectLighting(float3 position, GeometryPrimitiveSample primitive_sample) {
    return float3(0.0f, 0.0f, 0.0f);  // Return zero contribution
  }
};
