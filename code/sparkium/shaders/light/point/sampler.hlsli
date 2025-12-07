#pragma once
#include "bindings.hlsli"
#include "common.hlsli"

void PointLightSampler(inout SampleDirectLightingPayload payload) {
  float3 position = asfloat(payload.low.xyz);
  uint sampler_data_index = payload.low.w;
  ByteAddressBuffer direct_lighting_sampler_data = data_buffers[NonUniformResourceIndex(sampler_data_index)];

  float3 light_position = LoadFloat3(direct_lighting_sampler_data, 0);
  float3 light_emission = LoadFloat3(direct_lighting_sampler_data, 12);

  float3 omega_in = light_position - position;
  float shadow_length = length(omega_in);

  float3 eval = 1e6 * light_emission;
  float pdf = 1e6 * dot(omega_in, omega_in) * 4.0 * PI;
  omega_in = normalize(omega_in);

  payload.low.xyz = asuint(eval);
  payload.low.w = asuint(shadow_length);
  payload.high.xyz = asuint(omega_in);
  payload.high.w = asuint(pdf);
}
