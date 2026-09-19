#include "native_contract.hlsli"
#pragma once
#include "bindings.hlsli"
#include "common.hlsli"

void PointLightSampler(SP_CONTEXT inout SampleDirectLightingPayload payload) {
  float3 position = asfloat(payload.low.xyz);
  uint sampler_data_index = payload.low.w;
  ByteAddressBuffer direct_lighting_sampler_data = SP_BINDING_data_buffers[SP_NONUNIFORM(sampler_data_index)];

  float3 light_position = LoadFloat3(direct_lighting_sampler_data, 0);
  float3 light_power = LoadFloat3(direct_lighting_sampler_data, 12);
  float radius = asfloat(direct_lighting_sampler_data.Load(28));
  bool soft_falloff = direct_lighting_sampler_data.Load(32) != 0;

  float3 to_center = light_position - position;
  float distance_squared = dot(to_center, to_center);
  float distance = sqrt(distance_squared);
  float3 omega_in = to_center / max(distance, EPSILON);
  float shadow_length = distance;
  float3 eval;
  float pdf;

  if (radius <= EPSILON) {
    // Keep the delta-light convention used by the existing integrator.
    eval = 1e6f * light_power;
    pdf = 1e6f * distance_squared * 4.0f * PI;
  } else {
    float2 random_sample = asfloat(payload.high.yz);
    float radius_squared = radius * radius;
    if (soft_falloff) {
      // Blender's Soft Falloff uses an ad-hoc disk centered on the point light
      // and oriented toward each shading point (Cycles point.h).
      float3 light_normal = -omega_in;
      float3 tangent, bitangent;
      MakeOrthonormals(light_normal, tangent, bitangent);
      float disk_radius = radius * sqrt(random_sample.x);
      float phi = 2.0f * PI * random_sample.y;
      float3 sampled_position = light_position + disk_radius * (cos(phi) * tangent + sin(phi) * bitangent);
      float3 to_sample = sampled_position - position;
      shadow_length = length(to_sample);
      omega_in = to_sample / max(shadow_length, EPSILON);
      float light_cosine = max(dot(light_normal, -omega_in), EPSILON);
      pdf = shadow_length * shadow_length / (PI * radius_squared * light_cosine);
    } else if (distance_squared > radius_squared) {
      // Uniformly sample the solid angle subtended by the sphere. This matches
      // Cycles' spherical point light and avoids the large variance of sampling
      // its complete surface area.
      float cos_theta_max = sqrt(max(0.0f, 1.0f - radius_squared / distance_squared));
      float cos_theta = lerp(1.0f, cos_theta_max, random_sample.x);
      float sin_theta = sqrt(max(0.0f, 1.0f - cos_theta * cos_theta));
      float phi = 2.0f * PI * random_sample.y;
      float3 tangent, bitangent;
      MakeOrthonormals(omega_in, tangent, bitangent);
      omega_in =
          normalize(tangent * (cos(phi) * sin_theta) + bitangent * (sin(phi) * sin_theta) + omega_in * cos_theta);
      pdf = 1.0f / (2.0f * PI * max(1.0f - cos_theta_max, EPSILON));
    } else {
      // From inside the source, Cycles samples the receiving surface's cosine
      // hemisphere. The conceptual point light emits inward here even though a
      // literal emissive mesh would not.
      float3 surface_normal = normalize(asfloat(payload.extra.xyz));
      sample_cos_hemisphere(surface_normal, random_sample.x, random_sample.y, omega_in, pdf);
    }

    if (!soft_falloff) {
      float center_projection = dot(to_center, omega_in);
      float discriminant = max(0.0f, center_projection * center_projection + radius_squared - distance_squared);
      shadow_length =
          center_projection + (distance_squared > radius_squared ? -sqrt(discriminant) : sqrt(discriminant));
    }
    // Blender's normalized point-light power is converted to sphere radiance.
    eval = light_power / (4.0f * PI * PI * radius_squared);
    // Dedicated point lights are not part of Sparkium's acceleration
    // structure, so a BSDF-sampled path cannot hit them. Scale both terms to
    // keep eval/pdf unchanged while making the direct-light MIS weight one.
    eval *= 1e6f;
    pdf *= 1e6f;
  }

  payload.low.xyz = asuint(eval);
  payload.low.w = asuint(shadow_length);
  payload.high.xyz = asuint(omega_in);
  payload.high.w = asuint(pdf);
}
