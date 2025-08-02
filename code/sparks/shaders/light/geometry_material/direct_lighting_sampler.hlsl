#include "bindings.hlsli"
#include "common.hlsli"

// clang-format off
#include "geometry_sampler.hlsli"
#include "material_evaluator.hlsli"
// clang-format off

// callable shader to sample direct lighting
[shader("callable")] void SampleDirectLightingCallable(inout SampleDirectLightingPayload payload) {
  float3 position = asfloat(payload.low.xyz);
  uint sampler_data_index = payload.low.w;
  uint custom_index = payload.high.w;
  InstanceMetadata instance_meta = instance_metadatas[custom_index];
  float3 rv = asfloat(payload.high.xyz);
  ByteAddressBuffer direct_lighting_sampler_data = data_buffers[sampler_data_index];
  ByteAddressBuffer geometry_data = data_buffers[instance_meta.geometry_data_index];
  ByteAddressBuffer material_data = data_buffers[instance_meta.material_data_index];

  float3x4 transform = LoadFloat3x4(direct_lighting_sampler_data, 0);
  uint primitive_count = direct_lighting_sampler_data.Load(48);
  BufferReference<ByteAddressBuffer> power_cdf = MakeBufferReference(direct_lighting_sampler_data, 52);
  float total_power = power_cdf.Load(primitive_count * 4 - 4);
  GeometrySampler<ByteAddressBuffer> geometry_sampler;
  geometry_sampler.geometry_data = geometry_data;
  geometry_sampler.SetTransform(transform);
  MaterialEvaluator<ByteAddressBuffer> material_evaluator;
  material_evaluator.material_data = material_data;

  uint L = 0, R = primitive_count - 1;
  float r1 = rv.x;
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

  GeometryPrimitiveSample primitive_sample = geometry_sampler.SamplePrimitive(L, rv.yz);

  float3 omega_in = primitive_sample.position - position;
  float shadow_length = length(omega_in);

  float3 eval = material_evaluator.EvaluateDirectLighting(position, primitive_sample);
  float pdf = primitive_sample.pdf * dot(omega_in, omega_in) * prob;
  omega_in = normalize(omega_in);
  float NdotL = abs(dot(primitive_sample.normal, omega_in));
  pdf /= NdotL;

  payload.low.xyz = asuint(eval);
  payload.low.w = asuint(shadow_length);
  payload.high.xyz = asuint(omega_in);
  payload.high.w = asuint(pdf);
}
