#include "compute_contract.hlsli"
#pragma once
#include "common.hlsli"

SP_BUFFER_TEMPLATE
SP_CLASS MaterialEvaluator {
  SP_BUFFER_TYPE material_data;

  SP_GEOMETRY_TEMPLATE
  float PrimitivePower(SP_GEOMETRY_TYPE geometry_sampler, uint primitive_id) {
    float3 emission = LoadFloat3(material_data, 0);
    return max(emission.x, max(emission.y, emission.z)) * geometry_sampler.PrimitiveArea(primitive_id) * PI * 2.0f;
  }

  float3 EvaluateDirectLighting(float3 position, GeometryPrimitiveSample primitive_sample) {
    return LoadFloat3(material_data, 0);
  }
};
