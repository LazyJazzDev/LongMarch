#include "native_contract.hlsli"
#pragma once

SP_BUFFER_TEMPLATE
float3 MaterialLambertianEvaluateDirectLighting(SP_BUFFER_TYPE material_data,
                                                float3 position,
                                                GeometryPrimitiveSample primitive_sample) {
  float3 emission = LoadFloat3(material_data, 12);
  return emission;
}
