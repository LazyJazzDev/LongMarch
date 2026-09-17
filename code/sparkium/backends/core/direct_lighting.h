#pragma once
// Portable port of
//   code/sparkium/shaders/direct_lighting.hlsli
//   code/sparkium/shaders/light/point/sampler.hlsli
//   code/sparkium/shaders/light/geometry_material/sampler.hlsli
//   code/sparkium/shaders/geometry_primitive_sampler.hlsli
//   code/sparkium/shaders/material/*/eval_direct_light.hlsli

#include "sparkium/backends/core/buffer.h"
#include "sparkium/backends/core/geometry.h"
#include "sparkium/backends/core/rng.h"
#include "sparkium/backends/core/sampling.h"
#include "sparkium/backends/core/structs.h"

namespace sparkium::backends {

SPARKIUM_HD inline float PowerHeuristic(float base, float ref) {
  if (ref < SPARKIUM_EPSILON) {
    return 1.0f;  // Avoid division by zero
  }
  return (base * base) / (base * base + ref * ref);
}

// Port of SamplePrimitivePower (geometry_primitive_sampler.hlsli).
SPARKIUM_HD inline void SamplePrimitivePower(const float *power_cdf,
                                             uint32_t primitive_count,
                                             float &r,
                                             uint32_t &primitive_id,
                                             float &prob) {
  float total_power = power_cdf[primitive_count - 1];

  uint32_t L = 0, R = primitive_count - 1;
  while (L < R) {
    uint32_t mid = (L + R) / 2;
    float mid_power = power_cdf[mid];
    if (r <= mid_power / total_power) {
      R = mid;
    } else {
      L = mid + 1;
    }
  }

  primitive_id = L;
  float high_prob = power_cdf[L] / total_power;
  float low_prob = (L > 0) ? power_cdf[L - 1] / total_power : 0.0f;
  prob = high_prob - low_prob;

  r = (r - low_prob) / prob;
}

// Port of EvaluatePrimitiveProbability (geometry_primitive_sampler.hlsli).
SPARKIUM_HD inline float EvaluatePrimitiveProbability(const float *power_cdf,
                                                      uint32_t primitive_count,
                                                      uint32_t primitive_id) {
  float total_power = power_cdf[primitive_count - 1];
  float high_prob = power_cdf[primitive_id];
  float low_prob = (primitive_id > 0) ? power_cdf[primitive_id - 1] : 0.0f;
  return (high_prob - low_prob) / total_power;
}

// Port of MaterialLightEvaluateDirectLighting.
SPARKIUM_HD inline float3 MaterialLightEvaluateDirectLighting(const MaterialData &material_data,
                                                              const float3 &position,
                                                              const GeometryPrimitiveSample &primitive_sample) {
  float3 emission = material_data.emission;
  uint32_t two_sided = static_cast<uint32_t>(material_data.two_sided);
  float falloff_distance = material_data.falloff_distance;
  float3 omega_in = device::normalize(primitive_sample.position - position);
  if (two_sided != 0 || device::dot(primitive_sample.normal, omega_in) < 0.0f) {
    if (falloff_distance > 0.0f)
      emission *= device::saturate(1.0f - device::length(primitive_sample.position - position) / falloff_distance);
    // If two-sided or front-facing, return the emission
    return emission;
  }
  return float3(0.0f, 0.0f, 0.0f);  // If back-facing, return zero contribution
}

// Port of MaterialLambertianEvaluateDirectLighting.
SPARKIUM_HD inline float3 MaterialLambertianEvaluateDirectLighting(
    const MaterialData &material_data,
    const float3 & /*position*/,
    const GeometryPrimitiveSample & /*primitive_sample*/) {
  return material_data.emission;
}

// Port of MaterialPrincipledEvaluateDirectLighting.
SPARKIUM_HD inline float3 MaterialPrincipledEvaluateDirectLighting(
    const MaterialData &material_data,
    const float3 & /*position*/,
    const GeometryPrimitiveSample & /*primitive_sample*/) {
  return material_data.principled.emission_color * material_data.principled.emission_strength;
}

// Port of PointLightSampler.
SPARKIUM_HD inline void PointLightSampler(const LightData &light_data, SampleDirectLightingPayload &payload) {
  float3 position = payload.position;

  float3 light_position = light_data.light_position;
  float3 light_power = light_data.light_power;
  float radius = light_data.radius;
  bool soft_falloff = light_data.soft_falloff != 0;

  float3 to_center = light_position - position;
  float distance_squared = device::dot(to_center, to_center);
  float distance = sqrtf(distance_squared);
  float3 omega_in = to_center / device::max(distance, SPARKIUM_EPSILON);
  float shadow_length = distance;
  float3 eval;
  float pdf;

  if (radius <= SPARKIUM_EPSILON) {
    // Keep the delta-light convention used by the existing integrator.
    eval = 1e6f * light_power;
    pdf = 1e6f * distance_squared * 4.0f * SPARKIUM_PI;
  } else {
    float2 random_sample = float2(payload.sample.y, payload.sample.z);
    float radius_squared = radius * radius;
    if (soft_falloff) {
      // Blender's Soft Falloff uses an ad-hoc disk centered on the point light
      // and oriented toward each shading point (Cycles point.h).
      float3 light_normal = -omega_in;
      float3 tangent, bitangent;
      MakeOrthonormals(light_normal, tangent, bitangent);
      float disk_radius = radius * sqrtf(random_sample.x);
      float phi = 2.0f * SPARKIUM_PI * random_sample.y;
      float3 sampled_position = light_position + disk_radius * (cosf(phi) * tangent + sinf(phi) * bitangent);
      float3 to_sample = sampled_position - position;
      shadow_length = device::length(to_sample);
      omega_in = to_sample / device::max(shadow_length, SPARKIUM_EPSILON);
      float light_cosine = device::max(device::dot(light_normal, -omega_in), SPARKIUM_EPSILON);
      pdf = shadow_length * shadow_length / (SPARKIUM_PI * radius_squared * light_cosine);
    } else if (distance_squared > radius_squared) {
      // Uniformly sample the solid angle subtended by the sphere. This matches
      // Cycles' spherical point light and avoids the large variance of sampling
      // its complete surface area.
      float cos_theta_max = sqrtf(device::max(0.0f, 1.0f - radius_squared / distance_squared));
      float cos_theta = device::lerp(1.0f, cos_theta_max, random_sample.x);
      float sin_theta = sqrtf(device::max(0.0f, 1.0f - cos_theta * cos_theta));
      float phi = 2.0f * SPARKIUM_PI * random_sample.y;
      float3 tangent, bitangent;
      MakeOrthonormals(omega_in, tangent, bitangent);
      omega_in = device::normalize(tangent * (cosf(phi) * sin_theta) + bitangent * (sinf(phi) * sin_theta) +
                                   omega_in * cos_theta);
      pdf = 1.0f / (2.0f * SPARKIUM_PI * device::max(1.0f - cos_theta_max, SPARKIUM_EPSILON));
    } else {
      // From inside the source, Cycles samples the receiving surface's cosine
      // hemisphere. The conceptual point light emits inward here even though a
      // literal emissive mesh would not.
      float3 surface_normal = device::normalize(payload.normal);
      sample_cos_hemisphere(surface_normal, random_sample.x, random_sample.y, omega_in, pdf);
    }

    if (!soft_falloff) {
      float center_projection = device::dot(to_center, omega_in);
      float discriminant =
          device::max(0.0f, center_projection * center_projection + radius_squared - distance_squared);
      shadow_length = center_projection +
                      (distance_squared > radius_squared ? -sqrtf(discriminant) : sqrtf(discriminant));
    }
    // Blender's normalized point-light power is converted to sphere radiance.
    eval = light_power / (4.0f * SPARKIUM_PI * SPARKIUM_PI * radius_squared);
    // Dedicated point lights are not part of Sparkium's acceleration
    // structure, so a BSDF-sampled path cannot hit them. Scale both terms to
    // keep eval/pdf unchanged while making the direct-light MIS weight one.
    eval *= 1e6f;
    pdf *= 1e6f;
  }

  payload.eval = eval;
  payload.shadow_length = shadow_length;
  payload.omega_in = omega_in;
  payload.pdf = pdf;
}

// Port of MeshLightSampler.
SPARKIUM_HD inline void MeshLightSampler(const DeviceScene &scene,
                                         int32_t shader_index,
                                         const LightData &light_data,
                                         SampleDirectLightingPayload &payload) {
  float3 position = payload.position;
  uint32_t custom_index = static_cast<uint32_t>(light_data.custom_index);
  const InstanceData &instance_meta = scene.instances[custom_index];
  float3 rv = payload.sample;
  const MaterialData &material_data = scene.materials[instance_meta.material];
  const float *power_cdf = scene.primitive_power_cdf + light_data.primitive_cdf_offset;

  uint32_t primitive_id;
  float prob;
  float r = rv.x;
  SamplePrimitivePower(power_cdf, light_data.primitive_count, r, primitive_id, prob);

  GeometryPrimitiveSample primitive_sample = MeshSamplePrimitive(
      MeshBuffer(scene, instance_meta.mesh), light_data.mesh_transform, primitive_id, float2(rv.y, rv.z));

  float3 omega_in = primitive_sample.position - position;
  float shadow_length = device::length(omega_in);

  float3 eval = float3(0.0f, 0.0f, 0.0f);
  switch (shader_index) {
    case LIGHT_SAMPLER_MESH_LIGHT:
      eval = MaterialLightEvaluateDirectLighting(material_data, position, primitive_sample);
      break;
    case LIGHT_SAMPLER_MESH_LAMBERTIAN:
      eval = MaterialLambertianEvaluateDirectLighting(material_data, position, primitive_sample);
      break;
    case LIGHT_SAMPLER_MESH_SHADER_GRAPH:
      // Shader-graph emitters store their base color as the first float3 of the
      // material data; their surface shader itself is unavailable offline.
      eval = material_data.base_color;
      break;
    case LIGHT_SAMPLER_MESH_SPECULAR:
      eval = float3(0, 0, 0);
      break;
    case LIGHT_SAMPLER_MESH_PRINCIPLED:
      eval = MaterialPrincipledEvaluateDirectLighting(material_data, position, primitive_sample);
      break;
    default:
      break;
  }

  float pdf = primitive_sample.pdf * device::dot(omega_in, omega_in) * prob;
  omega_in = device::normalize(omega_in);
  float NdotL = fabsf(device::dot(primitive_sample.normal, omega_in));
  pdf /= NdotL;

  payload.eval = eval;
  payload.shadow_length = shadow_length;
  payload.omega_in = omega_in;
  payload.pdf = pdf;
}

// Port of DirectLightingProbability.
SPARKIUM_HD inline float DirectLightingProbability(const DeviceScene &scene, uint32_t light_index) {
  uint32_t light_count = scene.num_lights;
  if (light_index >= light_count) {
    return 0.0f;
  }
  float total_power = scene.light_power_cdf[light_count - 1];
  if (!(total_power > 0.0f))
    return 0.0f;
  float high_prob = scene.light_power_cdf[light_index] / total_power;
  float low_prob = (light_index > 0) ? scene.light_power_cdf[light_index - 1] / total_power : 0.0f;
  return high_prob - low_prob;
}

// Port of SampleDirectLighting (direct_lighting.hlsli).
SPARKIUM_HD inline void SampleDirectLighting(RenderContext &context,
                                            const HitRecord &hit_record,
                                            float3 &eval,
                                            float3 &omega_in,
                                            float &pdf,
                                            const DeviceScene &scene) {
  uint32_t light_count = scene.num_lights;
  eval = omega_in = float3(0, 0, 0);
  pdf = 0.0f;
  context.shadow_eval = context.shadow_dir = float3(0, 0, 0);
  context.shadow_length = 0.0f;
  if (light_count == 0)
    return;
  const float *power_cdf = scene.light_power_cdf;
  float total_power = power_cdf[light_count - 1];
  if (!(total_power > 0.0f))
    return;
  uint32_t L = 0, R = light_count - 1;
  float r1 = RandomFloat(context.rd, scene.sobol_table);
  while (L < R) {
    uint32_t mid = (L + R) / 2;
    float mid_power = power_cdf[mid];
    if (r1 <= mid_power / total_power) {
      R = mid;
    } else {
      L = mid + 1;
    }
  }
  float high_prob = power_cdf[L] / total_power;
  float low_prob = (L > 0) ? power_cdf[L - 1] / total_power : 0.0f;
  float prob = high_prob - low_prob;
  if (prob > SPARKIUM_EPSILON) {
    r1 = (r1 - low_prob) / prob;
  } else {
    r1 = 0.0f;  // Avoid division by zero
  }

  const LightData &light_meta = scene.lights[L];
  SampleDirectLightingPayload payload;
  payload.position = hit_record.position;
  // SampleDirectLightingPayload.sample is consumed as x/y/z by the light
  // samplers, so the two remaining uniforms have to advance in that order.
  float r2 = RandomFloat(context.rd, scene.sobol_table);
  float r3 = RandomFloat(context.rd, scene.sobol_table);
  payload.sample = float3(r1, r2, r3);
  payload.normal = hit_record.normal;
  payload.custom_index = light_meta.custom_index;
  payload.eval = float3(0, 0, 0);
  payload.shadow_length = 0.0f;
  payload.omega_in = float3(0, 0, 0);
  payload.pdf = 0.0f;
  if (light_meta.sampler_shader_index == LIGHT_SAMPLER_POINT)
    PointLightSampler(light_meta, payload);
  else
    MeshLightSampler(scene, light_meta.sampler_shader_index, light_meta, payload);

  eval = payload.eval;
  float shadow_length = payload.shadow_length * 0.9999;
  omega_in = payload.omega_in;
  pdf = payload.pdf * prob;
  context.shadow_eval = eval;
  context.shadow_dir = omega_in;
  context.shadow_length = shadow_length;
}

}  // namespace sparkium::backends
