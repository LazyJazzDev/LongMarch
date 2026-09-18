#pragma once

// Ports of `direct_lighting.hlsli`, `light/point/sampler.hlsli`,
// `light/geometry_material/sampler.hlsli` and the per-material
// `eval_direct_light.hlsli` helpers. The light selector CDF, the per-primitive
// power CDF and the sampler data blobs use the byte layouts the graphics
// pipelines upload, so light selection follows the same code path.

#include "native_principled.h"

namespace sparkium::native {

// Light sampler shader indices hard-coded by `LightSampler`.
#define SPARKIUM_LIGHT_SAMPLER_POINT 0x1000000
#define SPARKIUM_LIGHT_SAMPLER_MESH_LIGHT 0x1000001
#define SPARKIUM_LIGHT_SAMPLER_MESH_LAMBERTIAN 0x1000002
#define SPARKIUM_LIGHT_SAMPLER_MESH_PRINCIPLED 0x1000003
#define SPARKIUM_LIGHT_SAMPLER_MESH_SHADER_GRAPH 0x1000004
#define SPARKIUM_LIGHT_SAMPLER_MESH_SPECULAR 0x1000005

LM_DEVICE_FUNC inline float PowerHeuristic(float base, float ref) {
  if (ref < EPSILON) {
    return 1.0f;  // Avoid division by zero
  }
  return (base * base) / (base * base + ref * ref);
}

// ---------------------------------------------------------------------------
// material/*/eval_direct_light.hlsli
// ---------------------------------------------------------------------------
LM_DEVICE_FUNC inline float3 MaterialLightEvaluateDirectLighting(const ByteBuffer &material_data,
                                                                 const float3 &position,
                                                                 const GeometryPrimitiveSample &primitive_sample) {
  float3 emission = LoadFloat3(material_data, 0);
  const uint32_t two_sided = material_data.Load(12);
  const float falloff_distance = asfloat(material_data.Load(24));
  const float3 omega_in = glm::normalize(primitive_sample.position - position);
  if (two_sided || glm::dot(primitive_sample.normal, omega_in) < 0.0f) {
    if (falloff_distance > 0.0f)
      emission *= saturatef(1.0f - glm::length(primitive_sample.position - position) / falloff_distance);
    // If two-sided or front-facing, return the emission
    return emission;
  }
  return float3{0.0f, 0.0f, 0.0f};  // If back-facing, return zero contribution
}

LM_DEVICE_FUNC inline float3 MaterialLambertianEvaluateDirectLighting(const ByteBuffer &material_data,
                                                                      const float3 &position,
                                                                      const GeometryPrimitiveSample &sample) {
  return LoadFloat3(material_data, 12);
}

LM_DEVICE_FUNC inline float3 MaterialPrincipledEvaluateDirectLighting(const ByteBuffer &material_data,
                                                                      const float3 &position,
                                                                      const GeometryPrimitiveSample &sample) {
  const float4 emission = LoadFloat4(material_data, 92);
  return float3{emission.x, emission.y, emission.z} * emission.w;  // Scale by the emission intensity
}

// ---------------------------------------------------------------------------
// The HLSL light samplers communicate through a packed payload. The ported
// version keeps the same in/out fields so the arithmetic stays identical.
// ---------------------------------------------------------------------------
struct SampleDirectLightingPayload {
  // Inputs.
  float3 position;
  int sampler_data_index;
  float3 random;
  int custom_index;
  float3 normal;
  int ray_type;
  // Outputs.
  float3 eval;
  float shadow_length;
  float3 omega_in;
  float pdf;
};

LM_DEVICE_FUNC inline void PointLightSampler(const SceneView &scene, SampleDirectLightingPayload &payload) {
  const float3 position = payload.position;
  const ByteBuffer direct_lighting_sampler_data = scene.DataBuffer(payload.sampler_data_index);

  const float3 light_position = LoadFloat3(direct_lighting_sampler_data, 0);
  const float3 light_power = LoadFloat3(direct_lighting_sampler_data, 12);
  const float radius = asfloat(direct_lighting_sampler_data.Load(28));
  const bool soft_falloff = direct_lighting_sampler_data.Load(32) != 0;

  const float3 to_center = light_position - position;
  const float distance_squared = glm::dot(to_center, to_center);
  const float distance = ::sqrtf(distance_squared);
  float3 omega_in = to_center / ::fmaxf(distance, EPSILON);
  float shadow_length = distance;
  float3 eval;
  float pdf;

  if (radius <= EPSILON) {
    // Keep the delta-light convention used by the existing integrator.
    eval = 1e6f * light_power;
    pdf = 1e6f * distance_squared * 4.0f * PI;
  } else {
    const float2 random_sample{payload.random.y, payload.random.z};
    const float radius_squared = radius * radius;
    if (soft_falloff) {
      // Blender's Soft Falloff uses an ad-hoc disk centered on the point light
      // and oriented toward each shading point (Cycles point.h).
      const float3 light_normal = -omega_in;
      float3 tangent, bitangent;
      MakeOrthonormals(light_normal, tangent, bitangent);
      const float disk_radius = radius * ::sqrtf(random_sample.x);
      const float phi = 2.0f * PI * random_sample.y;
      const float3 sampled_position =
          light_position + disk_radius * (::cosf(phi) * tangent + ::sinf(phi) * bitangent);
      const float3 to_sample = sampled_position - position;
      shadow_length = glm::length(to_sample);
      omega_in = to_sample / ::fmaxf(shadow_length, EPSILON);
      const float light_cosine = ::fmaxf(glm::dot(light_normal, -omega_in), EPSILON);
      pdf = shadow_length * shadow_length / (PI * radius_squared * light_cosine);
    } else if (distance_squared > radius_squared) {
      // Uniformly sample the solid angle subtended by the sphere. This matches
      // Cycles' spherical point light and avoids the large variance of sampling
      // its complete surface area.
      const float cos_theta_max = ::sqrtf(::fmaxf(0.0f, 1.0f - radius_squared / distance_squared));
      const float cos_theta = lerp(1.0f, cos_theta_max, random_sample.x);
      const float sin_theta = ::sqrtf(::fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
      const float phi = 2.0f * PI * random_sample.y;
      float3 tangent, bitangent;
      MakeOrthonormals(omega_in, tangent, bitangent);
      omega_in = glm::normalize(tangent * (::cosf(phi) * sin_theta) + bitangent * (::sinf(phi) * sin_theta) +
                                omega_in * cos_theta);
      pdf = 1.0f / (2.0f * PI * ::fmaxf(1.0f - cos_theta_max, EPSILON));
    } else {
      // From inside the source, Cycles samples the receiving surface's cosine
      // hemisphere. The conceptual point light emits inward here even though a
      // literal emissive mesh would not.
      const float3 surface_normal = glm::normalize(payload.normal);
      sample_cos_hemisphere(surface_normal, random_sample.x, random_sample.y, omega_in, pdf);
    }

    if (!soft_falloff) {
      const float center_projection = glm::dot(to_center, omega_in);
      const float discriminant =
          ::fmaxf(0.0f, center_projection * center_projection + radius_squared - distance_squared);
      shadow_length = center_projection + (distance_squared > radius_squared ? -::sqrtf(discriminant)
                                                                            : ::sqrtf(discriminant));
    }
    // Blender's normalized point-light power is converted to sphere radiance.
    eval = light_power / (4.0f * PI * PI * radius_squared);
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

LM_DEVICE_FUNC inline void MeshLightSampler(const SceneView &scene,
                                            int shader_index,
                                            SampleDirectLightingPayload &payload) {
  const float3 position = payload.position;
  const InstanceMetadata instance_meta = scene.GetInstanceMetadata(payload.custom_index);
  const float3 rv = payload.random;
  const ByteBuffer direct_lighting_sampler_data = scene.DataBuffer(payload.sampler_data_index);
  const ByteBuffer geometry_data = scene.DataBuffer(instance_meta.geometry_data_index);
  const ByteBuffer material_data = scene.DataBuffer(instance_meta.material_data_index);

  const float3x4 transform = LoadFloat3x4(direct_lighting_sampler_data, 0);

  uint32_t primitive_id = 0;
  float prob = 0.0f;
  float r = rv.x;
  SamplePrimitivePower(direct_lighting_sampler_data, r, primitive_id, prob);

  const GeometryPrimitiveSample primitive_sample =
      MeshSamplePrimitive(geometry_data, transform, primitive_id, float2{rv.y, rv.z});

  float3 omega_in = primitive_sample.position - position;
  const float shadow_length = glm::length(omega_in);

  float3 eval{0.0f, 0.0f, 0.0f};
  switch (shader_index) {
    case SPARKIUM_LIGHT_SAMPLER_MESH_LIGHT:
      eval = MaterialLightEvaluateDirectLighting(material_data, position, primitive_sample);
      break;
    case SPARKIUM_LIGHT_SAMPLER_MESH_LAMBERTIAN:
      eval = MaterialLambertianEvaluateDirectLighting(material_data, position, primitive_sample);
      break;
    case SPARKIUM_LIGHT_SAMPLER_MESH_SHADER_GRAPH:
      eval = LoadFloat3(material_data, 0);
      break;
    case SPARKIUM_LIGHT_SAMPLER_MESH_SPECULAR:
      eval = float3{0, 0, 0};
      break;
    case SPARKIUM_LIGHT_SAMPLER_MESH_PRINCIPLED:
      eval = MaterialPrincipledEvaluateDirectLighting(material_data, position, primitive_sample);
      break;
    default:
      break;
  }

  float pdf = primitive_sample.pdf * glm::dot(omega_in, omega_in) * prob;
  omega_in = glm::normalize(omega_in);
  const float NdotL = ::fabsf(glm::dot(primitive_sample.normal, omega_in));
  pdf /= NdotL;

  payload.eval = eval;
  payload.shadow_length = shadow_length;
  payload.omega_in = omega_in;
  payload.pdf = pdf;
}

LM_DEVICE_FUNC inline void LightSampler(const SceneView &scene,
                                        int shader_index,
                                        SampleDirectLightingPayload &payload) {
  switch (shader_index) {
    case SPARKIUM_LIGHT_SAMPLER_POINT:
      PointLightSampler(scene, payload);
      break;
    case SPARKIUM_LIGHT_SAMPLER_MESH_LIGHT:
    case SPARKIUM_LIGHT_SAMPLER_MESH_LAMBERTIAN:
    case SPARKIUM_LIGHT_SAMPLER_MESH_PRINCIPLED:
    case SPARKIUM_LIGHT_SAMPLER_MESH_SHADER_GRAPH:
    case SPARKIUM_LIGHT_SAMPLER_MESH_SPECULAR:
      MeshLightSampler(scene, shader_index, payload);
      break;
    default:
      // The software path zeroes the payload for light types without a
      // hard-coded sampler; only mesh and point lights reach the native
      // backends because compute tracing requires triangle geometry.
      payload.eval = float3{0, 0, 0};
      payload.shadow_length = 0.0f;
      payload.omega_in = float3{0, 0, 0};
      payload.pdf = 0.0f;
      break;
  }
}

LM_DEVICE_FUNC inline void SampleDirectLighting(const SceneView &scene,
                                                RenderContext &context,
                                                const HitRecord &hit_record,
                                                float3 &eval,
                                                float3 &omega_in,
                                                float &pdf) {
  const uint32_t light_count = scene.light_selector_data.Load(0);
  eval = omega_in = float3{0, 0, 0};
  pdf = 0.0f;
  context.shadow_eval = context.shadow_dir = float3{0, 0, 0};
  context.shadow_length = 0.0f;
  if (light_count == 0)
    return;
  const BufferReference power_cdf = MakeBufferReference(scene.light_selector_data, 4);
  const float total_power = asfloat(power_cdf.Load(light_count * 4 - 4));
  if (!(total_power > 0.0f))
    return;
  uint32_t L = 0, R = light_count - 1;
  float r1 = RandomFloat(scene, context.rd);
  while (L < R) {
    const uint32_t mid = (L + R) / 2;
    const float mid_power = asfloat(power_cdf.Load(mid * 4));
    if (r1 <= mid_power / total_power) {
      R = mid;
    } else {
      L = mid + 1;
    }
  }
  const float high_prob = asfloat(power_cdf.Load(L * 4)) / total_power;
  const float low_prob = (L > 0) ? asfloat(power_cdf.Load((L - 1) * 4)) / total_power : 0.0f;
  const float prob = high_prob - low_prob;
  if (prob > EPSILON) {
    r1 = (r1 - low_prob) / prob;
  } else {
    r1 = 0.0f;  // Avoid division by zero
  }

  const LightMetadata light_meta = scene.GetLightMetadata(static_cast<int>(L));
  SampleDirectLightingPayload payload;
  payload.position = hit_record.position;
  payload.sampler_data_index = light_meta.sampler_data_index;
  payload.random = float3{r1, RandomFloat(scene, context.rd), RandomFloat(scene, context.rd)};
  payload.custom_index = light_meta.custom_index;
  payload.normal = hit_record.normal;
  payload.ray_type = context.ray_type;
  LightSampler(scene, light_meta.sampler_shader_index, payload);
  eval = payload.eval;
  const float shadow_length = payload.shadow_length * 0.9999f;
  omega_in = payload.omega_in;
  pdf = payload.pdf * prob;
  context.shadow_eval = eval;
  context.shadow_dir = omega_in;
  context.shadow_length = shadow_length;
}

LM_DEVICE_FUNC inline float DirectLightingProbability(const SceneView &scene, uint32_t light_index) {
  const uint32_t light_count = scene.light_selector_data.Load(0);
  if (light_index >= light_count) {
    return 0.0f;
  }
  const BufferReference power_cdf = MakeBufferReference(scene.light_selector_data, 4);
  const float total_power = asfloat(power_cdf.Load(light_count * 4 - 4));
  if (!(total_power > 0.0f))
    return 0.0f;
  const float high_prob = asfloat(power_cdf.Load(light_index * 4)) / total_power;
  const float low_prob = (light_index > 0) ? asfloat(power_cdf.Load((light_index - 1) * 4)) / total_power : 0.0f;
  return high_prob - low_prob;
}

// The emitter-hit MIS weight shared by every material sampler.
LM_DEVICE_FUNC inline float EmitterMISWeight(const SceneView &scene,
                                             const RenderContext &context,
                                             const HitRecord &hit_record,
                                             const InstanceMetadata &instance_meta,
                                             bool zero_on_grazing) {
  if (instance_meta.custom_index == -1)
    return 1.0f;
  const LightMetadata light_meta = scene.GetLightMetadata(instance_meta.custom_index);
  float pdf = hit_record.pdf * EvaluatePrimitiveProbability(scene.DataBuffer(light_meta.sampler_data_index),
                                                            static_cast<uint32_t>(hit_record.primitive_index));
  pdf *= DirectLightingProbability(scene, static_cast<uint32_t>(instance_meta.custom_index));
  const float3 omega_in = hit_record.position - context.origin;
  pdf *= glm::dot(omega_in, omega_in);
  const float NdotL = ::fabsf(glm::dot(hit_record.geom_normal, glm::normalize(omega_in)));
  if (zero_on_grazing && NdotL < EPSILON)
    return 0.0f;
  pdf /= NdotL;
  return PowerHeuristic(context.bsdf_pdf, pdf);
}

}  // namespace sparkium::native
