#include "native_contract.hlsli"
#pragma once

SP_BUFFER_TEMPLATE
float3 MaterialPrincipledEvaluateDirectLighting(SP_BUFFER_TYPE material_data,
                                                float3 position,
                                                GeometryPrimitiveSample primitive_sample) {
  float4 emission = LoadFloat4(material_data, 92);
  return emission.xyz * emission.w;  // Scale by the emission intensity
}
