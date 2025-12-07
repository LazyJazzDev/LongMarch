#pragma once
#include "bindings.hlsli"
#include "common.hlsli"
#include "geometry/mesh/sample_primitive.hlsli"
#include "material/light/eval_direct_light.hlsli"
#include "material/lambertian/eval_direct_light.hlsli"
#include "material/principled/eval_direct_light.hlsli"
#include "geometry_primitive_sampler.hlsli"

void MeshLightSampler(int shader_index, inout SampleDirectLightingPayload payload) {
  float3 position = asfloat(payload.low.xyz);
  uint sampler_data_index = payload.low.w;
  uint custom_index = payload.high.w;
  InstanceMetadata instance_meta = instance_metadatas.Load<InstanceMetadata>( sizeof(InstanceMetadata) * custom_index);
  float3 rv = asfloat(payload.high.xyz);
  ByteAddressBuffer direct_lighting_sampler_data = data_buffers[NonUniformResourceIndex(sampler_data_index)];
  ByteAddressBuffer geometry_data = data_buffers[NonUniformResourceIndex(instance_meta.geometry_data_index)];
  ByteAddressBuffer material_data = data_buffers[NonUniformResourceIndex(instance_meta.material_data_index)];

  float3x4 transform = LoadFloat3x4(direct_lighting_sampler_data, 0);

  uint primitive_id;
  float prob;
  SamplePrimitivePower(direct_lighting_sampler_data, rv.x, primitive_id, prob);

  GeometryPrimitiveSample primitive_sample = MeshSamplePrimitive(geometry_data, transform, primitive_id, rv.yz);

  float3 omega_in = primitive_sample.position - position;
  float shadow_length = length(omega_in);

  float3 eval;
  switch (shader_index) {
    case 0x1000001:
      eval = MaterialLightEvaluateDirectLighting(material_data, position, primitive_sample);
      break;
    case 0x1000002:
      eval = MaterialLambertianEvaluateDirectLighting(material_data, position, primitive_sample);
      break;
    case 0x1000003:
      eval = MaterialPrincipledEvaluateDirectLighting(material_data, position, primitive_sample);
      break;
  }

  float pdf = primitive_sample.pdf * dot(omega_in, omega_in) * prob;
  omega_in = normalize(omega_in);
  float NdotL = abs(dot(primitive_sample.normal, omega_in));
  pdf /= NdotL;

  payload.low.xyz = asuint(eval);
  payload.low.w = asuint(shadow_length);
  payload.high.xyz = asuint(omega_in);
  payload.high.w = asuint(pdf);
}
