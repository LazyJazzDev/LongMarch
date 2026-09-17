#pragma once
// Portable port of the per-material samplers and of the software shadow rays:
//   code/sparkium/shaders/material/lambertian/sampler.hlsl
//   code/sparkium/shaders/material/specular/sampler.hlsl
//   code/sparkium/shaders/material/light/sampler.hlsl
//   code/sparkium/shaders/material/principled/sampler.hlsl
//   code/sparkium/shaders/software/shadow.hlsli
//
// The four samplers share the emission next-event-estimation block; it is
// factored into EmissionMisWeight here instead of being repeated, and the
// online backends keep generating the equivalent HLSL dispatch from the same
// shader sources (see raytracing/software_pipeline.cpp).

#include "sparkium/backends/core/bsdf_principled.h"
#include "sparkium/backends/core/bsdf_basic.h"
#include "sparkium/backends/core/direct_lighting.h"
#include "sparkium/backends/core/geometry.h"
#include "sparkium/backends/core/texture.h"

namespace sparkium::backends {

using device::asfloat;
using device::asuint;

// ---------------------------------------------------------------------------
// Shadow rays (software/shadow.hlsli)
// ---------------------------------------------------------------------------

SPARKIUM_HD inline float ShadowRayNoAlpha(const DeviceScene &scene,
                                         const float3 &origin,
                                         const float3 &direction,
                                         float dist) {
  RayDesc ray;
  ray.Origin = origin;
  ray.Direction = direction;
  ray.TMin = SPARKIUM_T_MIN * device::max(device::length(origin), 1.0f);
  ray.TMax = dist;
  SoftwareHit hit;
  return InlineIntersect(scene, ray, true, hit) ? 0.0f : 1.0f;
}

// `SoftwareShadowTransmission`: materials with SAMPLE_SHADOW_NO_HITRECORD are
// fully opaque, blocker emitters are opaque and non-blocking emitters are
// transparent so traversal can find occluders behind them.
SPARKIUM_HD inline float ShadowTransmission(const DeviceScene &scene,
                                           uint32_t material_index,
                                           const HitRecord &hit_record,
                                           const float3 & /*direction*/) {
  const MaterialData &material = scene.materials[material_index];
  if (material.kind == MATERIAL_KIND_LIGHT)
    return material.block_ray != 0 ? 0.0f : 1.0f;
  return 0.0f;
}

SPARKIUM_HD inline float ShadowRay(const DeviceScene &scene,
                                  const float3 &origin,
                                  const float3 &direction,
                                  float dist) {
  RayDesc ray;
  ray.Origin = origin;
  ray.Direction = direction;
  ray.TMin = SPARKIUM_T_MIN * device::max(device::length(origin), 1.0f);
  ray.TMax = dist;
  float transmission = 1.0f;
  SoftwareHit hit;
  while (transmission > 1.0e-4f && InlineIntersect(scene, ray, false, hit)) {
    uint32_t material = scene.instances[hit.instance].material;
    HitRecord hit_record =
        MakeMeshHitRecord(scene.instances[hit.instance].mesh, hit.instance, hit.primitive, hit.barycentric,
                          hit.distance, direction, scene.instances[hit.instance].object_to_world,
                          scene.instances[hit.instance].world_to_object, scene);
    transmission *= ShadowTransmission(scene, material, hit_record, direction);
    // Advance one representable positive ray parameter, preserving close
    // transparent layers.
    ray.TMin = asfloat(asuint(hit.distance) + 1u);
  }
  return transmission;
}

// ---------------------------------------------------------------------------
// Emission MIS weight shared by the lambertian, light and principled samplers
// ---------------------------------------------------------------------------

SPARKIUM_HD inline float EmissionMisWeight(const DeviceScene &scene,
                                          const RenderContext &context,
                                          const HitRecord &hit_record,
                                          int32_t light_index) {
  if (light_index == -1)
    return 1.0f;
  const LightData &light_meta = scene.lights[static_cast<uint32_t>(light_index)];
  float pdf = hit_record.pdf * EvaluatePrimitiveProbability(scene.primitive_power_cdf + light_meta.primitive_cdf_offset,
                                                            light_meta.primitive_count,
                                                            static_cast<uint32_t>(hit_record.primitive_index));
  pdf *= DirectLightingProbability(scene, static_cast<uint32_t>(light_index));
  float3 omega_in = hit_record.position - context.origin;
  pdf *= device::dot(omega_in, omega_in);
  float NdotL = fabsf(device::dot(hit_record.geom_normal, device::normalize(omega_in)));
  pdf /= NdotL;
  return PowerHeuristic(context.bsdf_pdf, pdf);
}

// The principled sampler guards NdotL before dividing.
SPARKIUM_HD inline float EmissionMisWeightPrincipled(const DeviceScene &scene,
                                                     const RenderContext &context,
                                                     const HitRecord &hit_record,
                                                     int32_t light_index) {
  if (light_index == -1)
    return 1.0f;
  const LightData &light_meta = scene.lights[static_cast<uint32_t>(light_index)];
  float pdf = hit_record.pdf * EvaluatePrimitiveProbability(scene.primitive_power_cdf + light_meta.primitive_cdf_offset,
                                                            light_meta.primitive_count,
                                                            static_cast<uint32_t>(hit_record.primitive_index));
  pdf *= DirectLightingProbability(scene, static_cast<uint32_t>(light_index));
  float3 omega_in = hit_record.position - context.origin;
  pdf *= device::dot(omega_in, omega_in);
  float NdotL = fabsf(device::dot(hit_record.geom_normal, device::normalize(omega_in)));
  if (NdotL < SPARKIUM_EPSILON)
    return 0.0f;
  pdf /= NdotL;
  return PowerHeuristic(context.bsdf_pdf, pdf);
}

// ---------------------------------------------------------------------------
// Materials
// ---------------------------------------------------------------------------

SPARKIUM_HD inline void SampleMaterialLambertian(const DeviceScene &scene,
                                                 const MaterialData &material,
                                                 RenderContext &context,
                                                 HitRecord &hit_record) {
  float3 color = material.base_color;
  float3 emission = material.emission;

  {
    float3 eval;
    float3 omega_in;
    float pdf;
    SampleDirectLighting(context, hit_record, eval, omega_in, pdf, scene);
    float bsdf_pdf;
    float3 bsdf_eval = EvalLambertianBSDF(color, hit_record.normal, omega_in, bsdf_pdf);
    float mis_weight = PowerHeuristic(pdf, bsdf_pdf);
    if (pdf > SPARKIUM_EPSILON)
      context.shadow_eval = mis_weight * (eval / pdf) * bsdf_eval * context.throughput;
  }

  if (device::max(emission.x, device::max(emission.y, emission.z)) > 0.0f) {
    float mis_weight = EmissionMisWeight(scene, context, hit_record, scene.instances[hit_record.object_index].light);
    context.radiance += emission * context.throughput * mis_weight;
  }

  float3 eval;
  float3 omega_in;
  float pdf;
  SampleLambertianBSDF(color, context.rd, hit_record, eval, omega_in, pdf, scene.sobol_table);
  if (pdf < SPARKIUM_EPSILON) {
    context.throughput = float3(0, 0, 0);
    return;
  }
  context.throughput *= eval / pdf;
  context.origin = hit_record.position;
  context.direction = omega_in;
  context.bsdf_pdf = pdf;
  context.ray_type = SPARKIUM_RAY_TYPE_REFLECTION;
}

SPARKIUM_HD inline void SampleMaterialSpecular(const DeviceScene &scene,
                                               const MaterialData &material,
                                               RenderContext &context,
                                               HitRecord &hit_record) {
  float3 color = material.base_color;
  float3 eval;
  float3 omega_in;
  float pdf;
  SampleSpecularBSDF(color, context.direction, hit_record.normal, hit_record.geom_normal, eval, omega_in, pdf);
  context.throughput *= eval;
  context.origin = hit_record.position;
  context.direction = omega_in;
  context.bsdf_pdf = pdf;
  context.ray_type = SPARKIUM_RAY_TYPE_REFLECTION;
}

SPARKIUM_HD inline void SampleMaterialLight(const DeviceScene &scene,
                                            const MaterialData &material,
                                            RenderContext &context,
                                            HitRecord &hit_record) {
  float3 emission = material.emission;
  int32_t two_sided = material.two_sided;
  int32_t block_ray = material.block_ray;
  int32_t camera_visible = material.camera_visible;
  float falloff_distance = material.falloff_distance;
  if (falloff_distance > 0.0f)
    emission *= device::saturate(1.0f - hit_record.t / falloff_distance);
  // camera_visible only controls the primary camera path. A non-blocking
  // emitter does not scatter the ray, so crossing one or more coplanar light
  // triangles must keep camera visibility disabled. Actual reflection and
  // transmission events update ray_type and can see the emitter normally.
  if ((camera_visible != 0 || context.ray_type != SPARKIUM_RAY_TYPE_CAMERA) &&
      (two_sided != 0 || hit_record.front_facing)) {
    float mis_weight =
        EmissionMisWeight(scene, context, hit_record, scene.instances[hit_record.object_index].light);
    context.radiance += emission * context.throughput * mis_weight;
  }

  if (block_ray != 0) {
    context.throughput = float3(0.0f, 0.0f, 0.0f);
  }
  context.origin = hit_record.position;
}

SPARKIUM_HD inline void SampleMaterialPrincipled(const DeviceScene &scene,
                                                 const MaterialData &material,
                                                 RenderContext &context,
                                                 HitRecord &hit_record) {
  const PrincipledParams &params = material.principled;
  const uint8_t *material_bytes = reinterpret_cast<const uint8_t *>(&params);
  (void)material_bytes;

  PrincipledMaterial principled;
  principled.hit_record = hit_record;
  principled.omega_v = -context.direction;
  principled.base_color = params.base_color;
  principled.subsurface_color = params.subsurface_color;
  principled.subsurface = params.subsurface;
  principled.subsurface_radius = params.subsurface_radius;
  principled.metallic = params.metallic;
  principled.specular = params.specular;
  principled.specular_tint = params.specular_tint;
  principled.roughness = params.roughness;
  principled.anisotropic = params.anisotropic;
  principled.anisotropic_rotation = params.anisotropic_rotation;
  principled.sheen = params.sheen;
  principled.sheen_tint = params.sheen_tint;
  principled.clearcoat = params.clearcoat;
  principled.clearcoat_roughness = params.clearcoat_roughness;
  principled.ior = params.ior;
  principled.transmission = params.transmission;
  principled.transmission_roughness = params.transmission_roughness;

  float3 emission = params.emission_color;
  float strength = params.emission_strength;

  int32_t normal_texture_index = params.texture_index[PRINCIPLED_TEXTURE_NORMAL];
  float y_signal = params.normal_y_signal;
  if (normal_texture_index != -1 && fabsf(hit_record.signal) > 0.5f) {
    float3 tbn = SampleTexture(scene, normal_texture_index, hit_record.tex_coord).xyz() * 2.0f - 1.0f;
    float3x3 TBN = float3x3(hit_record.tangent, device::cross(hit_record.normal, hit_record.tangent) * y_signal,
                            hit_record.normal);
    principled.hit_record.normal = hit_record.normal = device::normalize(device::mul(tbn, TBN));
  }

  int32_t base_color_texture_index = params.texture_index[PRINCIPLED_TEXTURE_BASE_COLOR];
  if (base_color_texture_index != -1)
    principled.base_color = SampleTexture(scene, base_color_texture_index, hit_record.tex_coord).xyz();

  int32_t metallic_texture_index = params.texture_index[PRINCIPLED_TEXTURE_METALLIC];
  if (metallic_texture_index != -1)
    principled.metallic = SampleTexture(scene, metallic_texture_index, hit_record.tex_coord).x;

  int32_t specular_texture_index = params.texture_index[PRINCIPLED_TEXTURE_SPECULAR];
  if (specular_texture_index != -1)
    principled.specular = SampleTexture(scene, specular_texture_index, hit_record.tex_coord).x;

  int32_t roughness_texture_index = params.texture_index[PRINCIPLED_TEXTURE_ROUGHNESS];
  if (roughness_texture_index != -1)
    principled.roughness = SampleTexture(scene, roughness_texture_index, hit_record.tex_coord).x;

  int32_t anisotropic_texture_index = params.texture_index[PRINCIPLED_TEXTURE_ANISOTROPIC];
  if (anisotropic_texture_index != -1)
    principled.anisotropic = SampleTexture(scene, anisotropic_texture_index, hit_record.tex_coord).x;

  int32_t anisotropic_rotation_texture_index = params.texture_index[PRINCIPLED_TEXTURE_ANISOTROPIC_ROTATION];
  if (anisotropic_rotation_texture_index != -1)
    principled.anisotropic_rotation =
        SampleTexture(scene, anisotropic_rotation_texture_index, hit_record.tex_coord).x;

  int32_t emission_texture_index = params.texture_index[PRINCIPLED_TEXTURE_EMISSION];
  if (emission_texture_index != -1)
    emission *= SampleTexture(scene, emission_texture_index, hit_record.tex_coord).xyz();

  float3 eval;
  float3 omega_in;
  float pdf;

  {
    SampleDirectLighting(context, hit_record, eval, omega_in, pdf, scene);
    float bsdf_pdf;
    float3 bsdf_eval = principled.EvalPrincipledBSDF(omega_in, bsdf_pdf);
    float mis_weight = PowerHeuristic(pdf, bsdf_pdf);
    if (pdf > SPARKIUM_EPSILON && !device::isnanf(eval.x) && !device::isnanf(eval.y) && !device::isnanf(eval.z)) {
      eval /= pdf;
      context.shadow_eval = mis_weight * eval * bsdf_eval * context.throughput;
    }
  }

  emission *= strength;
  if (device::max(emission.x, device::max(emission.y, emission.z)) > 0.0f) {
    float mis_weight = EmissionMisWeightPrincipled(scene, context, hit_record,
                                                  scene.instances[hit_record.object_index].light);
    context.radiance += emission * context.throughput * mis_weight;
  }

  float bsdf_sample_u = RandomFloat(context.rd, scene.sobol_table);
  float bsdf_sample_v = RandomFloat(context.rd, scene.sobol_table);
  principled.SamplePrincipledBSDF(bsdf_sample_u, bsdf_sample_v, eval, omega_in, pdf);
  if (pdf < 1e-5) {
    context.throughput = float3(0, 0, 0);
  } else {
    context.throughput *= eval / pdf;
    context.origin = hit_record.position;
    context.direction = omega_in;
    context.bsdf_pdf = pdf;
    context.ray_type = device::dot(omega_in, hit_record.geom_normal) < 0.0f ? SPARKIUM_RAY_TYPE_TRANSMISSION
                                                                           : SPARKIUM_RAY_TYPE_REFLECTION;
  }
}

// Port of the generated `SoftwareSampleMaterial` dispatch.
SPARKIUM_HD inline void SampleMaterial(const DeviceScene &scene,
                                      uint32_t material_index,
                                      RenderContext &context,
                                      HitRecord &hit_record) {
  const MaterialData &material = scene.materials[material_index];
  switch (material.kind) {
    case MATERIAL_KIND_LAMBERTIAN:
      SampleMaterialLambertian(scene, material, context, hit_record);
      return;
    case MATERIAL_KIND_SPECULAR:
      SampleMaterialSpecular(scene, material, context, hit_record);
      return;
    case MATERIAL_KIND_LIGHT:
      SampleMaterialLight(scene, material, context, hit_record);
      return;
    case MATERIAL_KIND_PRINCIPLED:
      SampleMaterialPrincipled(scene, material, context, hit_record);
      return;
    default:
      context.throughput = float3(0, 0, 0);
      return;
  }
}

}  // namespace sparkium::backends
