#include "bindings.hlsli"
#include "common.hlsli"
#include "geometry_primitive_sampler.hlsli"

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

  uint primitive_id;
  float prob;
  SamplePrimitivePower(direct_lighting_sampler_data, rv.x, primitive_id, prob);

  GeometryPrimitiveSample primitive_sample = SamplePrimitive(geometry_data, transform, primitive_id, rv.yz);

  float3 omega_in = primitive_sample.position - position;
  float shadow_length = length(omega_in);

  float3 eval = EvaluateDirectLighting(material_data, position, primitive_sample);
  float pdf = primitive_sample.pdf * dot(omega_in, omega_in) * prob;
  omega_in = normalize(omega_in);
  float NdotL = abs(dot(primitive_sample.normal, omega_in));
  pdf /= NdotL;

  payload.low.xyz = asuint(eval);
  payload.low.w = asuint(shadow_length);
  payload.high.xyz = asuint(omega_in);
  payload.high.w = asuint(pdf);
}
