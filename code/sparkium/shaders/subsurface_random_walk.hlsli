#ifndef SPARKIUM_SHADER_SUBSURFACE_RANDOM_WALK_HLSLI_
#define SPARKIUM_SHADER_SUBSURFACE_RANDOM_WALK_HLSLI_

#include "bsdf/principled_util.hlsli"

float RandomWalkDistance(inout RenderContext context) {
  float sigma = max(context.medium_sigma_t[context.medium_channel], 1.0e-6f);
  return -log(max(1.0f - RandomFloat(context.rd), 1.0e-6f)) / sigma;
}

float3 SampleUniformSphere(inout RenderContext context) {
  float z = 1.0f - 2.0f * RandomFloat(context.rd);
  float phi = 2.0f * PI * RandomFloat(context.rd);
  float radius = sqrt(max(0.0f, 1.0f - z * z));
  return float3(radius * cos(phi), radius * sin(phi), z);
}

// Invert the diffuse reflectance of a semi-infinite random-walk medium to its
// single-scattering albedo. This is the Van de Hulst fit used by Cycles; base
// color is a target surface reflectance, not sigma_s / sigma_t directly.
float3 RandomWalkSingleScatteringAlbedo(float3 color) {
  color = clamp(color, 0.0f, 0.999999f);
  float3 s = 4.20863f * color -
             sqrt(9.59217f + 41.6808f * color +
                  17.7126f * color * color) +
             4.09712f;
  return clamp(1.0f - s * s, 0.0f, 0.999999f);
}

void StartSubsurfaceRandomWalk(inout RenderContext context,
                               HitRecord hit_record,
                               float3 albedo,
                               float3 radius,
                               float scale,
                               float ior) {
  float3 mean_free_path =
      max(radius * max(scale, 1.0e-4f), float3(1.0e-4f, 1.0e-4f, 1.0e-4f));
  context.medium_object_index = hit_record.object_index;
  context.medium_channel = min(int(RandomFloat(context.rd) * 3.0f), 2);
  context.medium_sigma_t = 1.0f / mean_free_path;
  context.medium_albedo = RandomWalkSingleScatteringAlbedo(albedo);
  context.medium_ior = max(ior, 1.0f);
  float3 channel_mask = float3(context.medium_channel == 0,
                               context.medium_channel == 1,
                               context.medium_channel == 2);
  context.throughput *= 3.0f * channel_mask;
  context.medium_sample_distance = RandomWalkDistance(context);
  context.origin = hit_record.position;
  // The caller has already sampled the diffuse-transmission surface interface.
  // Subsequent collisions sample the isotropic phase function.
  context.bsdf_pdf = INF;
  context.ray_type = RAY_TYPE_VOLUME;
}

bool ContinueSubsurfaceRandomWalk(inout RenderContext context,
                                  HitRecord hit_record) {
  if (context.medium_object_index < 0) return false;

  if (context.medium_sample_distance < hit_record.t) {
    // Analog sampling at one hero wavelength: the exponential free-flight
    // density and transmittance cancel, leaving only the scattering albedo.
    // Selecting the wavelength once at entry avoids multiplying RGB mixture
    // weights at every collision.
    context.throughput *= context.medium_albedo[context.medium_channel];
    context.origin += context.direction * context.medium_sample_distance;

    const float phase = 1.0f / (4.0f * PI);
    context.direction = SampleUniformSphere(context);
    context.bsdf_pdf = phase;
    context.ray_type = RAY_TYPE_VOLUME;
    context.medium_sample_distance = RandomWalkDistance(context);
    return true;
  }

  // Reaching the boundary is already sampled with the correct exponential
  // survival probability for the selected wavelength.
  int medium_object_index = context.medium_object_index;

  if (hit_record.object_index == medium_object_index) {
    float3 exit_normal = -hit_record.normal;

    // A path which reaches the boundary from inside still sees its Fresnel
    // interface. Reflected paths continue the random walk; only transmitted
    // paths proceed through the diffuse-transmission surface layer below.
    float interface_fresnel = fresnel_dielectric_cos(
        abs(dot(hit_record.normal, -context.direction)),
        1.0f / max(context.medium_ior, 1.0e-4f));
    if (RandomFloat(context.rd) < interface_fresnel) {
      context.direction = reflect(context.direction, hit_record.normal);
      context.origin = hit_record.position;
      context.bsdf_pdf = INF;
      context.ray_type = RAY_TYPE_VOLUME;
      context.medium_sample_distance = RandomWalkDistance(context);
      return true;
    }

    context.medium_object_index = -1;
    context.medium_sample_distance = INF;
    HitRecord exit_record = hit_record;
    exit_record.normal = exit_record.geom_normal = exit_normal;
    float3 light_eval, light_direction;
    float light_pdf;
    SampleDirectLighting(context, exit_record, light_eval, light_direction,
                         light_pdf);
    float cosine = max(dot(exit_normal, light_direction), 0.0f);
    if (light_pdf > EPSILON && all(isfinite(light_eval))) {
      context.shadow_eval = light_eval * (cosine / (PI * light_pdf)) *
                            context.throughput;
    }

    float direction_pdf;
    sample_cos_hemisphere(exit_normal, RandomFloat(context.rd),
                          RandomFloat(context.rd), context.direction,
                          direction_pdf);
    context.origin = hit_record.position;
    context.bsdf_pdf = direction_pdf;
    context.ray_type = RAY_TYPE_REFLECTION;
    return true;
  }
  context.medium_object_index = -1;
  context.medium_sample_distance = INF;
  return false;
}

#endif  // SPARKIUM_SHADER_SUBSURFACE_RANDOM_WALK_HLSLI_