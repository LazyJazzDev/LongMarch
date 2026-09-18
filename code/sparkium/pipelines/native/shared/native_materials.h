#pragma once

// Port of the five `material/*/sampler.hlsl` shaders plus the shader-graph
// surface layer (`material/shader_graph/surface_sampler.hlsli`).
//
// `SoftwarePipeline` resolves the material at a hit through a generated switch
// over per-material namespaces. The native backends do the same through
// `NativeMaterial::type`, which the scene flattener fills from the same
// `sparkium::Material` objects that drive shader generation.

#include "native_graph.h"
#include "native_subsurface.h"

namespace sparkium::native {

enum NativeMaterialType : uint32_t {
  NATIVE_MATERIAL_LAMBERTIAN = 0,
  NATIVE_MATERIAL_SPECULAR,
  NATIVE_MATERIAL_LIGHT,
  NATIVE_MATERIAL_PRINCIPLED,
  NATIVE_MATERIAL_SHADER_GRAPH
};

struct NativeMaterial {
  uint32_t type;
  int32_t graph_index;  // Index into `SceneView::graph_programs`, -1 otherwise.
};

// ---------------------------------------------------------------------------
// material/light/sampler.hlsl
// ---------------------------------------------------------------------------
LM_DEVICE_FUNC inline void SampleMaterialLight(const SceneView &scene,
                                               RenderContext &context,
                                               const HitRecord &hit_record) {
  const InstanceMetadata instance_meta = scene.GetInstanceMetadata(hit_record.object_index);
  const ByteBuffer material_buffer = scene.DataBuffer(instance_meta.material_data_index);
  float3 emission = LoadFloat3(material_buffer, 0);
  const int two_sided = static_cast<int>(material_buffer.Load(12));
  const int block_ray = static_cast<int>(material_buffer.Load(16));
  const int camera_visible = static_cast<int>(material_buffer.Load(20));
  const float falloff_distance = asfloat(material_buffer.Load(24));
  if (falloff_distance > 0.0f)
    emission *= saturatef(1.0f - hit_record.t / falloff_distance);
  // camera_visible only controls the primary camera path.  A non-blocking
  // emitter does not scatter the ray, so crossing one or more coplanar light
  // triangles must keep camera visibility disabled.  Actual reflection and
  // transmission events update ray_type and can see the emitter normally.
  if ((camera_visible || context.ray_type != RAY_TYPE_CAMERA) && (two_sided || hit_record.front_facing)) {
    const float mis_weight = EmitterMISWeight(scene, context, hit_record, instance_meta, false);
    context.radiance += emission * context.throughput * mis_weight;
  }

  if (block_ray) {
    context.throughput = float3{0.0f, 0.0f, 0.0f};
  }
  context.origin = hit_record.position;
}

LM_DEVICE_FUNC inline float ShadowOpacityLight(const SceneView &scene,
                                               const HitRecord &hit_record,
                                               const float3 &ray_direction) {
  const InstanceMetadata instance_meta = scene.GetInstanceMetadata(hit_record.object_index);
  const ByteBuffer material_buffer = scene.DataBuffer(instance_meta.material_data_index);
  return material_buffer.Load(16) ? 1.0f : 0.0f;
}

// ---------------------------------------------------------------------------
// material/lambertian/sampler.hlsl
// ---------------------------------------------------------------------------
LM_DEVICE_FUNC inline void SampleMaterialLambertian(const SceneView &scene,
                                                    RenderContext &context,
                                                    const HitRecord &hit_record) {
  const InstanceMetadata instance_meta = scene.GetInstanceMetadata(hit_record.object_index);
  const ByteBuffer material_buffer = scene.DataBuffer(instance_meta.material_data_index);
  const float3 color = LoadFloat3(material_buffer, 0);
  const float3 emission = LoadFloat3(material_buffer, 12);

  float3 eval;
  float3 omega_in;
  float pdf;
  {
    SampleDirectLighting(scene, context, hit_record, eval, omega_in, pdf);
    float bsdf_pdf;
    const float3 bsdf_eval = EvalLambertianBSDF(color, hit_record.normal, omega_in, bsdf_pdf);
    const float mis_weight = PowerHeuristic(pdf, bsdf_pdf);
    if (pdf > EPSILON)
      context.shadow_eval = mis_weight * (eval / pdf) * bsdf_eval * context.throughput;
  }

  if (max3(emission) > 0.0f) {
    const float mis_weight = EmitterMISWeight(scene, context, hit_record, instance_meta, false);
    context.radiance += emission * context.throughput * mis_weight;
  }

  SampleLambertianBSDF(scene, color, context.rd, hit_record, eval, omega_in, pdf);
  if (pdf < EPSILON) {
    context.throughput = float3{0, 0, 0};
    return;
  }
  context.throughput *= eval / pdf;
  context.origin = hit_record.position;
  context.direction = omega_in;
  context.bsdf_pdf = pdf;
  context.ray_type = RAY_TYPE_REFLECTION;
}

// ---------------------------------------------------------------------------
// material/specular/sampler.hlsl
// ---------------------------------------------------------------------------
LM_DEVICE_FUNC inline void SampleMaterialSpecular(const SceneView &scene,
                                                  RenderContext &context,
                                                  const HitRecord &hit_record) {
  const InstanceMetadata instance_meta = scene.GetInstanceMetadata(hit_record.object_index);
  const ByteBuffer material_buffer = scene.DataBuffer(instance_meta.material_data_index);
  const float3 color = LoadFloat3(material_buffer, 0);

  float3 eval;
  float3 omega_in;
  float pdf;

  SampleSpecularBSDF(color, context.direction, hit_record.normal, hit_record.geom_normal, eval, omega_in, pdf);
  context.throughput *= eval;
  context.origin = hit_record.position;
  context.direction = omega_in;
  context.bsdf_pdf = pdf;
  context.ray_type = RAY_TYPE_REFLECTION;
}

// ---------------------------------------------------------------------------
// material/principled/sampler.hlsl
// ---------------------------------------------------------------------------
LM_DEVICE_FUNC inline void SampleMaterialPrincipled(const SceneView &scene,
                                                    RenderContext &context,
                                                    HitRecord hit_record) {
  const InstanceMetadata instance_meta = scene.GetInstanceMetadata(hit_record.object_index);
  StreamedBufferReference material_buffer =
      MakeStreamedBufferReference(scene.DataBuffer(instance_meta.material_data_index), 0);

  PrincipledMaterial material;
  material.hit_record = hit_record;
  material.omega_v = -context.direction;
  material.base_color = material_buffer.LoadFloat3();
  material.subsurface_color = material_buffer.LoadFloat3();
  material.subsurface = material_buffer.LoadFloat();
  material.subsurface_radius = material_buffer.LoadFloat3();
  material.metallic = material_buffer.LoadFloat();
  material.specular = material_buffer.LoadFloat();
  material.specular_tint = material_buffer.LoadFloat();

  material.roughness = material_buffer.LoadFloat();
  material.anisotropic = material_buffer.LoadFloat();
  material.anisotropic_rotation = material_buffer.LoadFloat();
  material.sheen = material_buffer.LoadFloat();
  material.sheen_tint = material_buffer.LoadFloat();
  material.clearcoat = material_buffer.LoadFloat();
  material.clearcoat_roughness = material_buffer.LoadFloat();
  material.ior = material_buffer.LoadFloat();
  material.transmission = material_buffer.LoadFloat();
  material.transmission_roughness = material_buffer.LoadFloat();

  float3 emission = material_buffer.LoadFloat3();
  const float strength = material_buffer.LoadFloat();

  const int normal_texture_index = material_buffer.LoadInt();
  const float y_signal = material_buffer.LoadFloat();
  if (normal_texture_index != -1 && ::fabsf(hit_record.signal) > 0.5f) {
    const float4 sampled = SampleTexture(scene, normal_texture_index, hit_record.tex_coord);
    const float3 tbn = float3{sampled.x, sampled.y, sampled.z} * 2.0f - 1.0f;
    // HLSL: mul(tbn, float3x3(tangent, cross(normal, tangent) * y_signal, normal)).
    material.hit_record.normal = hit_record.normal = glm::normalize(mul_rows(
        tbn, hit_record.tangent, glm::cross(hit_record.normal, hit_record.tangent) * y_signal, hit_record.normal));
  }

  const int base_color_texture_index = material_buffer.LoadInt();
  if (base_color_texture_index != -1) {
    const float4 texel = SampleTexture(scene, base_color_texture_index, hit_record.tex_coord);
    material.base_color = float3{texel.x, texel.y, texel.z};
  }

  const int metallic_texture_index = material_buffer.LoadInt();
  if (metallic_texture_index != -1)
    material.metallic = SampleTexture(scene, metallic_texture_index, hit_record.tex_coord).x;

  const int specular_texture_index = material_buffer.LoadInt();
  if (specular_texture_index != -1)
    material.specular = SampleTexture(scene, specular_texture_index, hit_record.tex_coord).x;

  const int roughness_texture_index = material_buffer.LoadInt();
  if (roughness_texture_index != -1)
    material.roughness = SampleTexture(scene, roughness_texture_index, hit_record.tex_coord).x;

  const int anisotropic_texture_index = material_buffer.LoadInt();
  if (anisotropic_texture_index != -1)
    material.anisotropic = SampleTexture(scene, anisotropic_texture_index, hit_record.tex_coord).x;

  const int anisotropic_rotation_texture_index = material_buffer.LoadInt();
  if (anisotropic_rotation_texture_index != -1)
    material.anisotropic_rotation =
        SampleTexture(scene, anisotropic_rotation_texture_index, hit_record.tex_coord).x;

  const int emission_texture_index = material_buffer.LoadInt();
  if (emission_texture_index != -1) {
    const float4 texel = SampleTexture(scene, emission_texture_index, hit_record.tex_coord);
    emission *= float3{texel.x, texel.y, texel.z};
  }

  float3 eval;
  float3 omega_in;
  float pdf;

  {
    SampleDirectLighting(scene, context, hit_record, eval, omega_in, pdf);
    float bsdf_pdf;
    const float3 bsdf_eval = material.EvalPrincipledBSDF(omega_in, bsdf_pdf);
    const float mis_weight = PowerHeuristic(pdf, bsdf_pdf);
    if (pdf > EPSILON && !IsNaN(eval.x) && !IsNaN(eval.y) && !IsNaN(eval.z)) {
      eval /= pdf;
      context.shadow_eval = mis_weight * eval * bsdf_eval * context.throughput;
    }
  }

  emission *= strength;
  if (max3(emission) > 0.0f) {
    const float mis_weight = EmitterMISWeight(scene, context, hit_record, instance_meta, true);
    context.radiance += emission * context.throughput * mis_weight;
  }

  material.SamplePrincipledBSDF(RandomFloat(scene, context.rd), RandomFloat(scene, context.rd), eval, omega_in, pdf);
  if (pdf < 1e-5f) {
    context.throughput = float3{0, 0, 0};
  } else {
    context.throughput *= eval / pdf;
    context.origin = hit_record.position;
    context.direction = omega_in;
    context.bsdf_pdf = pdf;
    context.ray_type = glm::dot(omega_in, hit_record.geom_normal) < 0.0f ? RAY_TYPE_TRANSMISSION : RAY_TYPE_REFLECTION;
  }
}

// ---------------------------------------------------------------------------
// material/shader_graph/surface_sampler.hlsli
// ---------------------------------------------------------------------------
LM_DEVICE_FUNC inline void SampleGraphSurface(const SceneView &scene,
                                              RenderContext &context,
                                              HitRecord hit_record,
                                              const GraphSurface &graph) {
  // A Transparent BSDF continues the current path without changing direction.
  // Mixed closures use opacity as the probability of selecting the opaque branch.
  if (graph.opacity < 1.0f && RandomFloat(scene, context.rd) >= saturatef(graph.opacity)) {
    context.origin = hit_record.position;
    context.bsdf_pdf = INF;
    return;
  }

  PrincipledMaterial material;
  material.hit_record = hit_record;
  material.hit_record.normal = hit_record.normal = glm::normalize(graph.normal);
  material.omega_v = -context.direction;
  material.base_color = graph.base_color;
  material.subsurface_color = graph.base_color;
  const float3 random_walk_radius = graph.subsurface_radius * graph.subsurface_scale;
  const bool has_random_walk_radius = max3(random_walk_radius) >= 1.0e-4f;
  const bool use_random_walk = graph.subsurface_method > 0.5f && graph.subsurface > 0.0f && has_random_walk_radius;
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
    SampleDirectLighting(scene, context, hit_record, light_eval, light_direction, light_pdf);
    float reflection_pdf;
    const float3 reflection_eval = material.EvalPrincipledBSDF(light_direction, reflection_pdf);
    const float mis_weight = PowerHeuristic(light_pdf, reflection_pdf);
    if (light_pdf > EPSILON && AllFinite(light_eval)) {
      context.shadow_eval = mis_weight * light_eval / light_pdf * reflection_eval * context.throughput;
    }

    const float reflection_probability = material.PrincipledThinReflectionProbability();
    const float event_sample = RandomFloat(scene, context.rd);
    if (event_sample >= reflection_probability) {
      context.throughput *= graph.base_color;
      context.origin = hit_record.position;
      context.bsdf_pdf = INF;
      context.ray_type = RAY_TYPE_TRANSMISSION;
      return;
    }

    float3 eval, omega_in;
    float pdf;
    material.SamplePrincipledThinReflection(event_sample / ::fmaxf(reflection_probability, CLOSURE_WEIGHT_CUTOFF),
                                            RandomFloat(scene, context.rd), eval, omega_in, pdf);
    if (pdf < 1e-5f) {
      context.throughput = float3{0.0f, 0.0f, 0.0f};
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
    const float thin_mix = saturatef(graph.transmission) * (1.0f - saturatef(graph.metallic));
    PrincipledMaterial opaque_material = material;
    opaque_material.transmission = 0.0f;
    PrincipledMaterial glass_material = material;
    glass_material.metallic = 0.0f;
    glass_material.transmission = 1.0f;

    float3 light_eval, light_direction;
    float light_pdf;
    SampleDirectLighting(scene, context, hit_record, light_eval, light_direction, light_pdf);
    float opaque_pdf, glass_pdf;
    const float3 opaque_eval = opaque_material.EvalPrincipledBSDF(light_direction, opaque_pdf);
    const float3 glass_eval = glass_material.EvalPrincipledBSDF(light_direction, glass_pdf);
    const float combined_pdf = lerp(opaque_pdf, glass_pdf, thin_mix);
    const float3 combined_eval = lerp(opaque_eval, glass_eval, thin_mix);
    const float mis_weight = PowerHeuristic(light_pdf, combined_pdf);
    if (light_pdf > EPSILON && AllFinite(light_eval)) {
      context.shadow_eval = mis_weight * light_eval / light_pdf * combined_eval * context.throughput;
    }

    const float event_sample = RandomFloat(scene, context.rd);
    float3 sampled_direction{0.0f, 0.0f, 0.0f};
    float sampled_pdf = 0.0f;
    if (event_sample < thin_mix) {
      const float reflection_probability = glass_material.PrincipledThinReflectionProbability();
      const float glass_sample = event_sample / ::fmaxf(thin_mix, CLOSURE_WEIGHT_CUTOFF);
      if (glass_sample >= reflection_probability) {
        context.throughput *= graph.base_color;
        context.origin = hit_record.position;
        context.bsdf_pdf = INF;
        context.ray_type = RAY_TYPE_TRANSMISSION;
        return;
      }
      float3 sampled_eval;
      glass_material.SamplePrincipledThinReflection(
          glass_sample / ::fmaxf(reflection_probability, CLOSURE_WEIGHT_CUTOFF), RandomFloat(scene, context.rd),
          sampled_eval, sampled_direction, sampled_pdf);
    } else {
      float3 sampled_eval;
      opaque_material.SamplePrincipledBSDF((event_sample - thin_mix) / ::fmaxf(1.0f - thin_mix, CLOSURE_WEIGHT_CUTOFF),
                                           RandomFloat(scene, context.rd), sampled_eval, sampled_direction,
                                           sampled_pdf);
    }

    if (sampled_pdf < 1e-5f) {
      context.throughput = float3{0.0f, 0.0f, 0.0f};
      return;
    }

    float sampled_opaque_pdf, sampled_glass_pdf;
    const float3 sampled_opaque_eval = opaque_material.EvalPrincipledBSDF(sampled_direction, sampled_opaque_pdf);
    const float3 sampled_glass_eval = glass_material.EvalPrincipledBSDF(sampled_direction, sampled_glass_pdf);
    const float sampled_combined_pdf = lerp(sampled_opaque_pdf, sampled_glass_pdf, thin_mix);
    const float3 sampled_combined_eval = lerp(sampled_opaque_eval, sampled_glass_eval, thin_mix);
    if (sampled_combined_pdf < 1e-5f) {
      context.throughput = float3{0.0f, 0.0f, 0.0f};
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
    SampleDirectLighting(scene, context, hit_record, light_eval, light_direction, light_pdf);
    float surface_pdf;
    const float3 surface_eval = material.EvalPrincipledBSDF(light_direction, surface_pdf);
    const float mis_weight = PowerHeuristic(light_pdf, surface_pdf);
    if (light_pdf > EPSILON && AllFinite(light_eval)) {
      context.shadow_eval = mis_weight * light_eval / light_pdf * surface_eval * context.throughput;
    }

    const float interface_fresnel = fresnel_dielectric_cos(
        ::fabsf(glm::dot(glm::normalize(hit_record.normal), -context.direction)), ::fmaxf(graph.ior, 1.0e-4f));
    const float surface_probability =
        saturatef(interface_fresnel + graph.clearcoat * 0.04f * (1.0f - interface_fresnel));
    if (RandomFloat(scene, context.rd) < surface_probability) {
      float3 eval, reflected_direction;
      float pdf;
      material.SamplePrincipledBSDF(RandomFloat(scene, context.rd), RandomFloat(scene, context.rd), eval,
                                    reflected_direction, pdf);
      if (pdf < 1.0e-5f || surface_probability < 1.0e-5f) {
        context.throughput = float3{0.0f, 0.0f, 0.0f};
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
    sample_cos_hemisphere(-hit_record.normal, RandomFloat(scene, context.rd), RandomFloat(scene, context.rd),
                          diffuse_transmission, diffuse_pdf);
    RefractionBsdf entry_bsdf;
    entry_bsdf.weight = float3{1.0f, 1.0f, 1.0f};
    entry_bsdf.sample_weight = 1.0f;
    entry_bsdf.N = hit_record.normal;
    const float entry_roughness = graph.subsurface_method > 1.5f ? 1.0f : saturatef(graph.roughness);
    entry_bsdf.alpha = ::fmaxf(entry_roughness * entry_roughness, 1.0e-4f);
    entry_bsdf.ior = ::fmaxf(graph.ior, 1.0e-4f);
    float3 entry_eval{0.0f, 0.0f, 0.0f};
    float3 refracted_direction{0.0f, 0.0f, 0.0f};
    float entry_pdf = 0.0f;
    bsdf_microfacet_ggx_sample_refraction(entry_bsdf, hit_record.geom_normal, -context.direction,
                                          RandomFloat(scene, context.rd), RandomFloat(scene, context.rd), entry_eval,
                                          refracted_direction, entry_pdf);
    const bool diffuse_skin_entry = graph.subsurface_method > 1.5f && RandomFloat(scene, context.rd) < 0.5f;
    context.direction = (entry_pdf > 0.0f && !diffuse_skin_entry) ? refracted_direction : diffuse_transmission;
    context.throughput *= (1.0f - interface_fresnel) / ::fmaxf(1.0f - surface_probability, 1.0e-5f);
    StartSubsurfaceRandomWalk(scene, context, hit_record, graph.base_color, graph.subsurface_radius,
                              graph.subsurface_scale, graph.ior);
    return;
  }

  float3 eval, omega_in;
  float pdf;
  SampleDirectLighting(scene, context, hit_record, eval, omega_in, pdf);
  float bsdf_pdf;
  const float3 bsdf_eval = material.EvalPrincipledBSDF(omega_in, bsdf_pdf);
  const float mis_weight = PowerHeuristic(pdf, bsdf_pdf);
  if (pdf > EPSILON && AllFinite(eval))
    context.shadow_eval = mis_weight * eval / pdf * bsdf_eval * context.throughput;

  if (max3(graph.emission) > 0.0f)
    context.radiance += graph.emission * context.throughput;

  material.SamplePrincipledBSDF(RandomFloat(scene, context.rd), RandomFloat(scene, context.rd), eval, omega_in, pdf);
  if (pdf < 1e-5f) {
    context.throughput = float3{0.0f, 0.0f, 0.0f};
  } else {
    context.throughput *= eval / pdf;
    context.origin = hit_record.position;
    context.direction = omega_in;
    context.bsdf_pdf = pdf;
    context.ray_type = glm::dot(omega_in, hit_record.geom_normal) < 0.0f ? RAY_TYPE_TRANSMISSION : RAY_TYPE_REFLECTION;
  }
}

LM_DEVICE_FUNC inline float GraphShadowOpacity(const GraphSurface &graph) {
  const float random_walk_weight = graph.subsurface_method > 0.5f ? saturatef(graph.subsurface) : 0.0f;
  const float shadow_opacity = graph.shadow_opacity >= 0.0f ? graph.shadow_opacity : graph.opacity;
  return shadow_opacity * (1.0f - random_walk_weight);
}

// ---------------------------------------------------------------------------
// Material dispatch, mirroring the generated `SoftwareSampleMaterial` and
// `SoftwareShadowTransmission` switches.
// ---------------------------------------------------------------------------
LM_DEVICE_FUNC inline void SampleMaterial(const SceneView &scene,
                                          uint32_t material,
                                          RenderContext &context,
                                          const HitRecord &hit_record) {
  if (material >= scene.material_count) {
    context.throughput = float3{0, 0, 0};
    return;
  }
  const NativeMaterial native_material = scene.materials[material];
  switch (native_material.type) {
    case NATIVE_MATERIAL_LAMBERTIAN:
      SampleMaterialLambertian(scene, context, hit_record);
      break;
    case NATIVE_MATERIAL_SPECULAR:
      SampleMaterialSpecular(scene, context, hit_record);
      break;
    case NATIVE_MATERIAL_LIGHT:
      SampleMaterialLight(scene, context, hit_record);
      break;
    case NATIVE_MATERIAL_PRINCIPLED:
      SampleMaterialPrincipled(scene, context, hit_record);
      break;
    case NATIVE_MATERIAL_SHADER_GRAPH: {
      const InstanceMetadata instance_meta = scene.GetInstanceMetadata(hit_record.object_index);
      const ByteBuffer material_data = scene.DataBuffer(instance_meta.material_data_index);
      const GraphSurface graph =
          EvaluateShaderGraph(scene, scene.graph_programs[native_material.graph_index], material_data, hit_record,
                              -context.direction, context.bounce, context.ray_type, false);
      SampleGraphSurface(scene, context, hit_record, graph);
      break;
    }
    default:
      context.throughput = float3{0, 0, 0};
      break;
  }
}

// `Transmission()` in the generated software renderer: only light and
// shader-graph materials define SAMPLE_SHADOW_ANY_HIT; every other material
// returns the closest-hit payload default of 0.
LM_DEVICE_FUNC inline float ShadowTransmission(const SceneView &scene,
                                               uint32_t material,
                                               const HitRecord &hit_record,
                                               const float3 &direction) {
  if (material >= scene.material_count)
    return 0.0f;
  const NativeMaterial native_material = scene.materials[material];
  switch (native_material.type) {
    case NATIVE_MATERIAL_LIGHT:
      return 1.0f - saturatef(ShadowOpacityLight(scene, hit_record, direction));
    case NATIVE_MATERIAL_SHADER_GRAPH: {
      const InstanceMetadata instance_meta = scene.GetInstanceMetadata(hit_record.object_index);
      const ByteBuffer material_data = scene.DataBuffer(instance_meta.material_data_index);
      const GraphSurface graph =
          EvaluateShaderGraph(scene, scene.graph_programs[native_material.graph_index], material_data, hit_record,
                              -direction, 1, RAY_TYPE_REFLECTION, true);
      return 1.0f - saturatef(GraphShadowOpacity(graph));
    }
    default:
      return 0.0f;
  }
}

}  // namespace sparkium::native
