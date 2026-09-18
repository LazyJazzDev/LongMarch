#pragma once

// Port of `subsurface_random_walk.hlsli`.

#include "native_lighting.h"

namespace sparkium::native {

LM_DEVICE_FUNC inline float RandomWalkDistance(const SceneView &scene, RenderContext &context) {
  const float sigma = ::fmaxf(context.medium_sigma_t[context.medium_channel], 1.0e-6f);
  return -::logf(::fmaxf(1.0f - RandomFloat(scene, context.rd), 1.0e-6f)) / sigma;
}

LM_DEVICE_FUNC inline float3 SampleUniformSphere(const SceneView &scene, RenderContext &context) {
  const float z = 1.0f - 2.0f * RandomFloat(scene, context.rd);
  const float phi = 2.0f * PI * RandomFloat(scene, context.rd);
  const float radius = ::sqrtf(::fmaxf(0.0f, 1.0f - z * z));
  return float3{radius * ::cosf(phi), radius * ::sinf(phi), z};
}

// Invert the diffuse reflectance of a semi-infinite random-walk medium to its
// single-scattering albedo. This is the Van de Hulst fit used by Cycles; base
// color is a target surface reflectance, not sigma_s / sigma_t directly.
LM_DEVICE_FUNC inline float3 RandomWalkSingleScatteringAlbedo(float3 color) {
  color = glm::clamp(color, make_float3(0.0f), make_float3(0.999999f));
  const float3 inner = 9.59217f + 41.6808f * color + 17.7126f * color * color;
  const float3 s = 4.20863f * color -
                   float3{::sqrtf(inner.x), ::sqrtf(inner.y), ::sqrtf(inner.z)} + 4.09712f;
  return glm::clamp(1.0f - s * s, make_float3(0.0f), make_float3(0.999999f));
}

LM_DEVICE_FUNC inline void StartSubsurfaceRandomWalk(const SceneView &scene,
                                                     RenderContext &context,
                                                     const HitRecord &hit_record,
                                                     const float3 &albedo,
                                                     const float3 &radius,
                                                     float scale,
                                                     float ior) {
  const float3 mean_free_path = glm::max(radius * ::fmaxf(scale, 1.0e-4f), make_float3(1.0e-4f));
  context.medium_object_index = hit_record.object_index;
  context.medium_channel = glm::min(static_cast<int>(RandomFloat(scene, context.rd) * 3.0f), 2);
  context.medium_sigma_t = 1.0f / mean_free_path;
  context.medium_albedo = RandomWalkSingleScatteringAlbedo(albedo);
  context.medium_ior = ::fmaxf(ior, 1.0f);
  const float3 channel_mask{context.medium_channel == 0 ? 1.0f : 0.0f, context.medium_channel == 1 ? 1.0f : 0.0f,
                            context.medium_channel == 2 ? 1.0f : 0.0f};
  context.throughput *= 3.0f * channel_mask;
  context.medium_sample_distance = RandomWalkDistance(scene, context);
  context.origin = hit_record.position;
  // The caller has already sampled the diffuse-transmission surface interface.
  // Subsequent collisions sample the isotropic phase function.
  context.bsdf_pdf = INF;
  context.ray_type = RAY_TYPE_VOLUME;
}

LM_DEVICE_FUNC inline bool ContinueSubsurfaceRandomWalk(const SceneView &scene,
                                                        RenderContext &context,
                                                        const HitRecord &hit_record) {
  if (context.medium_object_index < 0)
    return false;

  if (context.medium_sample_distance < hit_record.t) {
    // Analog sampling at one hero wavelength: the exponential free-flight
    // density and transmittance cancel, leaving only the scattering albedo.
    // Selecting the wavelength once at entry avoids multiplying RGB mixture
    // weights at every collision.
    context.throughput *= context.medium_albedo[context.medium_channel];
    context.origin += context.direction * context.medium_sample_distance;

    const float phase = 1.0f / (4.0f * PI);
    context.direction = SampleUniformSphere(scene, context);
    context.bsdf_pdf = phase;
    context.ray_type = RAY_TYPE_VOLUME;
    context.medium_sample_distance = RandomWalkDistance(scene, context);
    return true;
  }

  // Reaching the boundary is already sampled with the correct exponential
  // survival probability for the selected wavelength.
  const int medium_object_index = context.medium_object_index;

  if (hit_record.object_index == medium_object_index) {
    const float3 exit_normal = -hit_record.normal;

    // A path which reaches the boundary from inside still sees its Fresnel
    // interface. Reflected paths continue the random walk; only transmitted
    // paths proceed through the diffuse-transmission surface layer below.
    const float interface_fresnel = fresnel_dielectric_cos(::fabsf(glm::dot(hit_record.normal, -context.direction)),
                                                           1.0f / ::fmaxf(context.medium_ior, 1.0e-4f));
    if (RandomFloat(scene, context.rd) < interface_fresnel) {
      context.direction = reflect(context.direction, hit_record.normal);
      context.origin = hit_record.position;
      context.bsdf_pdf = INF;
      context.ray_type = RAY_TYPE_VOLUME;
      context.medium_sample_distance = RandomWalkDistance(scene, context);
      return true;
    }

    context.medium_object_index = -1;
    context.medium_sample_distance = INF;
    HitRecord exit_record = hit_record;
    exit_record.normal = exit_record.geom_normal = exit_normal;
    float3 light_eval, light_direction;
    float light_pdf;
    SampleDirectLighting(scene, context, exit_record, light_eval, light_direction, light_pdf);
    const float cosine = ::fmaxf(glm::dot(exit_normal, light_direction), 0.0f);
    if (light_pdf > EPSILON && AllFinite(light_eval)) {
      context.shadow_eval = light_eval * (cosine / (PI * light_pdf)) * context.throughput;
    }

    float direction_pdf;
    sample_cos_hemisphere(exit_normal, RandomFloat(scene, context.rd), RandomFloat(scene, context.rd),
                          context.direction, direction_pdf);
    context.origin = hit_record.position;
    context.bsdf_pdf = direction_pdf;
    context.ray_type = RAY_TYPE_REFLECTION;
    return true;
  }
  context.medium_object_index = -1;
  context.medium_sample_distance = INF;
  return false;
}

}  // namespace sparkium::native
