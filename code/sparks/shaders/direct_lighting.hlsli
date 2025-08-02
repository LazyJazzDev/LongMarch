#pragma once
#include "bindings.hlsli"
#include "shadow_ray.hlsli"

void SampleDirectLighting(in RenderContext context, HitRecord hit_record, out float3 eval, out float3 omega_in, out float pdf) {
  uint light_count = light_selector_data.Load(0);
  BufferReference<ByteAddressBuffer> power_cdf = MakeBufferReference(light_selector_data, 4);
  float total_power = power_cdf.Load(light_count * 4 - 4);
  uint L = 0, R = light_count - 1;
  float r1 = RandomFloat(context.rd);
  while (L < R) {
    uint mid = (L + R) / 2;
    float mid_power = power_cdf.Load(mid * 4);
    if (r1 <= mid_power / total_power) {
      R = mid;
    } else {
      L = mid + 1;
    }
  }
  float high_prob = power_cdf.Load(L * 4) / total_power;
  float low_prob = (L > 0) ? power_cdf.Load((L - 1) * 4) / total_power : 0.0f;
  float prob = high_prob - low_prob;
  r1 = (r1 - low_prob) / prob;

  LightMetadata light_meta = light_metadatas[L];
  SampleDirectLightingPayload payload;
  payload.low.xyz = asuint(hit_record.position);
  payload.low.w = light_meta.sampler_data_index;
  payload.high.xyz = asuint(float3(r1, RandomFloat(context.rd), RandomFloat(context.rd)));
  payload.high.w = light_meta.custom_index;
  CallShader(light_meta.sampler_shader_index, payload);
  eval = asfloat(payload.low.xyz);
  float shadow_length = asfloat(payload.low.w);
  omega_in = asfloat(payload.high.xyz);
  pdf = asfloat(payload.high.w) * prob;
  eval *= ShadowRay(hit_record.position, omega_in, shadow_length);
}

float DirectLightingProbability(uint light_index) {
  uint light_count = light_selector_data.Load(0);
  if (light_index >= light_count) {
    return 0.0f;
  }
  BufferReference<ByteAddressBuffer> power_cdf = MakeBufferReference(light_selector_data, 4);
  float total_power = power_cdf.Load(light_count * 4 - 4);
  float high_prob = power_cdf.Load(light_index * 4) / total_power;
  float low_prob = (light_index > 0) ? power_cdf.Load((light_index - 1) * 4) / total_power : 0.0f;
  return high_prob - low_prob;
}

float PowerHeuristic(float base, float ref) {
  return (base * base) / (base * base + ref * ref);
}
