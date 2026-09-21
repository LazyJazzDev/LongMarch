#include "compute_contract.hlsli"
#pragma once
#include "common.hlsli"

SP_BUFFER_TEMPLATE
SP_CLASS MaterialEvaluator {
  SP_BUFFER_TYPE material_data;

  SP_GEOMETRY_TEMPLATE
  float PrimitivePower(SP_GEOMETRY_TYPE geometry_sampler, uint primitive_id) {
    return 0.0;  // Use max to get the maximum power
  }

  float3 EvaluateDirectLighting(float3 position, GeometryPrimitiveSample primitive_sample) {
    return float3(0.0f, 0.0f, 0.0f);  // Return zero contribution
  }
};
