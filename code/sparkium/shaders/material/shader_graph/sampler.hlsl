#pragma once
#include "bindings.hlsli"
#include "bsdf/principled_material.hlsli"
#include "buffer_helper.hlsli"
#include "direct_lighting.hlsli"
#include "geometry_primitive_sampler.hlsli"

#define SAMPLE_SHADOW_ANY_HIT

struct GraphSurface {
  float3 base_color;
  float metallic;
  float specular;
  float roughness;
  float anisotropic;
  float anisotropic_rotation;
  float sheen;
  float clearcoat;
  float clearcoat_roughness;
  float ior;
  float transmission;
  float transmission_roughness;
  float3 emission;
  float3 normal;
  float opacity;
  float shadow_opacity;
  float thin_walled;
  float subsurface;
  float subsurface_scale;
  float3 subsurface_radius;
  float subsurface_method;
};

// SHADER_GRAPH_IMPLEMENTATION

void SampleMaterial(inout RenderContext context, HitRecord hit_record) {
  InstanceMetadata instance_meta = instance_metadatas.Load<InstanceMetadata>(sizeof(InstanceMetadata) * hit_record.object_index);
  ByteAddressBuffer material_data = data_buffers[NonUniformResourceIndex(instance_meta.material_data_index)];
  GraphSurface graph = EvaluateShaderGraph(hit_record, -context.direction, context.bounce, context.ray_type, false,
                                           material_data);

  // A Transparent BSDF continues the current path without changing direction.
  // Mixed closures use opacity as the probability of selecting the opaque branch.
  if (graph.opacity < 1.0f && RandomFloat(context.rd) >= saturate(graph.opacity)) {
    context.origin = hit_record.position;
    context.bsdf_pdf = INF;
    return;
  }

  PrincipledMaterial material;
  material.hit_record = hit_record;
  material.hit_record.normal = hit_record.normal = normalize(graph.normal);
  material.omega_v = -context.direction;
  material.base_color = graph.base_color;
  material.subsurface_color = graph.base_color;
  float3 random_walk_radius = graph.subsurface_radius * graph.subsurface_scale;
  bool has_random_walk_radius =
      max(random_walk_radius.x,
          max(random_walk_radius.y, random_walk_radius.z)) >= 1.0e-4f;
  bool use_random_walk = graph.subsurface_method > 0.5f &&
                         graph.subsurface > 0.0f && has_random_walk_radius;
  material.subsurface = use_random_walk ? graph.subsurface : 0.0f;
  material.subsurface_radius = graph.subsurface_radius;
  material.metallic = graph.metallic;
  material.specular = graph.specular;
  material.specular_tint = 0.0f;
  material.roughness = graph.roughness;
  material.anisotropic = graph.anisotropic;
  material.anisotropic_rotation = graph.anisotropic_rotation;
  material.sheen = graph.sheen;
  material.sheen_tint = 0.0f;
  material.clearcoat = graph.clearcoat;
  material.clearcoat_roughness = graph.clearcoat_roughness;
  material.ior = graph.ior;
  material.transmission = graph.transmission;
  material.transmission_roughness = graph.transmission_roughness;

  // The benchmark glass keeps Fresnel reflection while transmission continues
  // in the incident direction.  Its reflection is still a rough GGX lobe and
  // therefore needs light sampling; treating it as a perfect mirror loses the
  // broad area-light highlights on bottles and bulb envelopes.
  if (graph.thin_walled > 1.5f && graph.transmission > 0.0f) {
    float3 light_eval, light_direction;
    float light_pdf;
    SampleDirectLighting(context, hit_record, light_eval, light_direction,
                         light_pdf);
    float reflection_pdf;
    float3 reflection_eval =
        material.EvalPrincipledBSDF(light_direction, reflection_pdf);
    float mis_weight = PowerHeuristic(light_pdf, reflection_pdf);
    if (light_pdf > EPSILON && all(isfinite(light_eval))) {
      context.shadow_eval = mis_weight * light_eval / light_pdf *
                            reflection_eval * context.throughput;
    }

    float reflection_probability =
        material.PrincipledThinReflectionProbability();
    float event_sample = RandomFloat(context.rd);
    if (event_sample >= reflection_probability) {
      context.throughput *= graph.base_color;
      context.origin = hit_record.position;
      context.bsdf_pdf = INF;
      context.ray_type = RAY_TYPE_TRANSMISSION;
      return;
    }

    float3 eval, omega_in;
    float pdf;
    material.SamplePrincipledThinReflection(
        event_sample / max(reflection_probability, CLOSURE_WEIGHT_CUTOFF),
        RandomFloat(context.rd), eval, omega_in, pdf);
    if (pdf < 1e-5f) {
      context.throughput = float3(0.0f, 0.0f, 0.0f);
    } else {
      context.throughput *= eval / pdf;
      context.origin = hit_record.position;
      context.direction = omega_in;
      context.bsdf_pdf = pdf;
      context.ray_type = RAY_TYPE_REFLECTION;
    }
    return;
  }

  // Some benchmark props encode thin glass by feeding an inverted opacity map
  // into Principled Transmission.  Treat that value as a spatial mixture of an
  // opaque Principled surface and thin dielectric glass.  The previous null-only
  // shortcut erased Fresnel highlights, making the central bottle disappear.
  if (graph.thin_walled > 0.5f && graph.transmission > 0.0f) {
    float thin_mix = saturate(graph.transmission) *
                     (1.0f - saturate(graph.metallic));
    PrincipledMaterial opaque_material = material;
    opaque_material.transmission = 0.0f;
    PrincipledMaterial glass_material = material;
    glass_material.metallic = 0.0f;
    glass_material.transmission = 1.0f;

    float3 light_eval, light_direction;
    float light_pdf;
    SampleDirectLighting(context, hit_record, light_eval, light_direction,
                         light_pdf);
    float opaque_pdf, glass_pdf;
    float3 opaque_eval =
        opaque_material.EvalPrincipledBSDF(light_direction, opaque_pdf);
    float3 glass_eval =
        glass_material.EvalPrincipledBSDF(light_direction, glass_pdf);
    float combined_pdf = lerp(opaque_pdf, glass_pdf, thin_mix);
    float3 combined_eval = lerp(opaque_eval, glass_eval, thin_mix);
    float mis_weight = PowerHeuristic(light_pdf, combined_pdf);
    if (light_pdf > EPSILON && all(isfinite(light_eval))) {
      context.shadow_eval = mis_weight * light_eval / light_pdf *
                            combined_eval * context.throughput;
    }

    float event_sample = RandomFloat(context.rd);
    float3 sampled_direction = float3(0.0f, 0.0f, 0.0f);
    float sampled_pdf = 0.0f;
    if (event_sample < thin_mix) {
      float reflection_probability =
          glass_material.PrincipledThinReflectionProbability();
      float glass_sample = event_sample / max(thin_mix, CLOSURE_WEIGHT_CUTOFF);
      if (glass_sample >= reflection_probability) {
        context.throughput *= graph.base_color;
        context.origin = hit_record.position;
        context.bsdf_pdf = INF;
        context.ray_type = RAY_TYPE_TRANSMISSION;
        return;
      }
      float3 sampled_eval;
      glass_material.SamplePrincipledThinReflection(
          glass_sample / max(reflection_probability, CLOSURE_WEIGHT_CUTOFF),
          RandomFloat(context.rd), sampled_eval, sampled_direction,
          sampled_pdf);
    } else {
      float3 sampled_eval;
      opaque_material.SamplePrincipledBSDF(
          (event_sample - thin_mix) /
              max(1.0f - thin_mix, CLOSURE_WEIGHT_CUTOFF),
          RandomFloat(context.rd), sampled_eval, sampled_direction,
          sampled_pdf);
    }

    if (sampled_pdf < 1e-5f) {
      context.throughput = float3(0.0f, 0.0f, 0.0f);
      return;
    }

    float sampled_opaque_pdf, sampled_glass_pdf;
    float3 sampled_opaque_eval = opaque_material.EvalPrincipledBSDF(
        sampled_direction, sampled_opaque_pdf);
    float3 sampled_glass_eval = glass_material.EvalPrincipledBSDF(
        sampled_direction, sampled_glass_pdf);
    float sampled_combined_pdf =
        lerp(sampled_opaque_pdf, sampled_glass_pdf, thin_mix);
    float3 sampled_combined_eval =
        lerp(sampled_opaque_eval, sampled_glass_eval, thin_mix);
    if (sampled_combined_pdf < 1e-5f) {
      context.throughput = float3(0.0f, 0.0f, 0.0f);
    } else {
      context.throughput *= sampled_combined_eval / sampled_combined_pdf;
      context.origin = hit_record.position;
      context.direction = sampled_direction;
      context.bsdf_pdf = sampled_combined_pdf;
      context.ray_type = RAY_TYPE_REFLECTION;
    }
    return;
  }

  if (use_random_walk && hit_record.front_facing) {
    // The surface dielectric/coat remains directly visible. Diffuse energy is
    // disabled in Principled while subsurface is active, so this evaluates only
    // the surface lobes before the path enters the random-walk medium.
    float3 light_eval, light_direction;
    float light_pdf;
    SampleDirectLighting(context, hit_record, light_eval, light_direction,
                         light_pdf);
    float surface_pdf;
    float3 surface_eval =
        material.EvalPrincipledBSDF(light_direction, surface_pdf);
    float mis_weight = PowerHeuristic(light_pdf, surface_pdf);
    if (light_pdf > EPSILON && all(isfinite(light_eval))) {
      context.shadow_eval = mis_weight * light_eval / light_pdf *
                            surface_eval * context.throughput;
    }

    float interface_fresnel = fresnel_dielectric_cos(
        abs(dot(normalize(hit_record.normal), -context.direction)),
        max(graph.ior, 1.0e-4f));
    float surface_probability = saturate(
        interface_fresnel + graph.clearcoat * 0.04f *
                                (1.0f - interface_fresnel));
    if (RandomFloat(context.rd) < surface_probability) {
      float3 eval, reflected_direction;
      float pdf;
      material.SamplePrincipledBSDF(RandomFloat(context.rd),
                                    RandomFloat(context.rd), eval,
                                    reflected_direction, pdf);
      if (pdf < 1.0e-5f || surface_probability < 1.0e-5f) {
        context.throughput = float3(0.0f, 0.0f, 0.0f);
      } else {
        context.throughput *= eval / (pdf * surface_probability);
        context.origin = hit_record.position;
        context.direction = reflected_direction;
        context.bsdf_pdf = pdf * surface_probability;
        context.ray_type = RAY_TYPE_REFLECTION;
      }
      return;
    }

    // Cycles RANDOM_WALK enters through a rough GGX transmissive boundary;
    // RANDOM_WALK_SKIN mixes that with a diffuse-transmission entry. Only the
    // resulting inward direction is needed here because the boundary energy
    // is already accounted for by the Fresnel branch above.
    float3 diffuse_transmission;
    float diffuse_pdf;
    sample_cos_hemisphere(-hit_record.normal, RandomFloat(context.rd),
                          RandomFloat(context.rd), diffuse_transmission,
                          diffuse_pdf);
    PrincipledMaterial::RefractionBsdf entry_bsdf;
    entry_bsdf.weight = float3(1.0f, 1.0f, 1.0f);
    entry_bsdf.sample_weight = 1.0f;
    entry_bsdf.N = hit_record.normal;
    float entry_roughness = graph.subsurface_method > 1.5f
                                ? 1.0f
                                : saturate(graph.roughness);
    entry_bsdf.alpha = max(entry_roughness * entry_roughness, 1.0e-4f);
    entry_bsdf.ior = max(graph.ior, 1.0e-4f);
    float3 entry_eval = float3(0.0f, 0.0f, 0.0f);
    float3 refracted_direction = float3(0.0f, 0.0f, 0.0f);
    float entry_pdf = 0.0f;
    material.bsdf_microfacet_ggx_sample_refraction(
        entry_bsdf, hit_record.geom_normal, -context.direction,
        RandomFloat(context.rd), RandomFloat(context.rd), entry_eval,
        refracted_direction, entry_pdf);
    bool diffuse_skin_entry = graph.subsurface_method > 1.5f &&
                              RandomFloat(context.rd) < 0.5f;
    context.direction = (entry_pdf > 0.0f && !diffuse_skin_entry)
                            ? refracted_direction
                            : diffuse_transmission;
    context.throughput *= (1.0f - interface_fresnel) /
                          max(1.0f - surface_probability, 1.0e-5f);
    StartSubsurfaceRandomWalk(context, hit_record, graph.base_color,
                              graph.subsurface_radius,
                              graph.subsurface_scale, graph.ior);
    return;
  }

  float3 eval, omega_in;
  float pdf;
  SampleDirectLighting(context, hit_record, eval, omega_in, pdf);
  float bsdf_pdf;
  float3 bsdf_eval = material.EvalPrincipledBSDF(omega_in, bsdf_pdf);
  float mis_weight = PowerHeuristic(pdf, bsdf_pdf);
  if (pdf > EPSILON && all(isfinite(eval))) context.shadow_eval = mis_weight * eval / pdf * bsdf_eval * context.throughput;

  if (max(graph.emission.x, max(graph.emission.y, graph.emission.z)) > 0.0f)
    context.radiance += graph.emission * context.throughput;

  material.SamplePrincipledBSDF(RandomFloat(context.rd), RandomFloat(context.rd), eval, omega_in, pdf);
  if (pdf < 1e-5f) {
    context.throughput = float3(0.0f, 0.0f, 0.0f);
  } else {
    context.throughput *= eval / pdf;
    context.origin = hit_record.position;
    context.direction = omega_in;
    context.bsdf_pdf = pdf;
    context.ray_type = dot(omega_in, hit_record.geom_normal) < 0.0f ? RAY_TYPE_TRANSMISSION : RAY_TYPE_REFLECTION;
  }
}

float SampleShadowOpacity(HitRecord hit_record) {
  InstanceMetadata instance_meta = instance_metadatas.Load<InstanceMetadata>(sizeof(InstanceMetadata) * hit_record.object_index);
  ByteAddressBuffer material_data = data_buffers[NonUniformResourceIndex(instance_meta.material_data_index)];
  GraphSurface graph = EvaluateShaderGraph(hit_record, -WorldRayDirection(), 1, RAY_TYPE_REFLECTION, true,
                                           material_data);
  float random_walk_weight = graph.subsurface_method > 0.5f
                                 ? saturate(graph.subsurface)
                                 : 0.0f;
  float shadow_opacity = graph.shadow_opacity >= 0.0f ? graph.shadow_opacity : graph.opacity;
  return shadow_opacity * (1.0f - random_walk_weight);
}

void SampleShadow(inout ShadowRayPayload payload, HitRecord hit_record) {
  // Opacity is accumulated by ShadowAnyHit. Reaching closest-hit means the
  // accumulated transmittance has become zero.
  payload.shadow = 0.0f;
}
