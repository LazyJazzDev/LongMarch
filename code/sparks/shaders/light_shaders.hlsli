#pragma once
#include "material_shaders.hlsli"

// Light Sampler Implementation

void LightSampler(int shader_index, inout SampleDirectLightingPayload payload) {

  switch (shader_index) {
// LightSampler Function List
  default:
    break;
  }

  payload.low.xyz = asuint(float3(0.0, 0.0, 0.0));
  payload.low.w = asuint(0.0);
  payload.high.xyz = asuint(float3(0.0, 1.0, 0.0));
  payload.high.w = asuint(1.0);
}
