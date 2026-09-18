// Material dispatch for the CPU backend.
//
// SoftwarePipeline::CompileRenderer builds this dispatch as HLSL text at
// runtime and hands it to DXC, one namespace per distinct material source. The
// CPU backend knows its material set at build time, so the equivalent C++ is
// emitted here instead: all five samplers are compiled in, and only the
// shader-graph evaluator is resolved at runtime, because that code is generated
// per scene (see graph_program.h).
//
// Scene data still decides which kernel a hit uses: materials are compacted by
// source text exactly as SoftwarePipeline does, and material_kernels maps each
// compacted index to the kernel the pipeline recognised.
#pragma once

#include "sparkium/pipelines/raytracing/cpu/shaders/hlsl_cpu_bindings.h"

namespace sparkium_cpu_shaders {

// Same prerequisites, in the same order, as software/render.hlsl uses before it
// reaches the material dispatch. The include guards make the repetition free.
#include "random.hlsli"
#include "direct_lighting.hlsli"
#include "subsurface_random_walk.hlsli"

// GraphSurface has to be complete before the graph sampler is compiled, since
// the evaluator entry point below returns it. surface_sampler.hlsli carries an
// include guard, so the sampler's own include of it is a no-op.
#include "material/shader_graph/surface_sampler.hlsli"

// Filled in by the pipeline once per material set. Indexed by the compacted
// material index the shaders receive.
enum MaterialKernel : uint32_t {
  MATERIAL_KERNEL_LAMBERTIAN = 0,
  MATERIAL_KERNEL_LIGHT = 1,
  MATERIAL_KERNEL_PRINCIPLED = 2,
  MATERIAL_KERNEL_SPECULAR = 3,
  MATERIAL_KERNEL_SHADER_GRAPH = 4,
};
inline std::vector<uint32_t> material_kernels;

// Implemented by the pipeline. The shader-graph sampler calls this with the
// same signature the generated per-material function has on the GPU; the
// pipeline resolves the hit to the material's compiled graph program.
GraphSurface EvaluateShaderGraph(HitRecord hit_record,
                                 float3 view_direction,
                                 int bounce,
                                 int ray_type,
                                 bool is_shadow_ray,
                                 ByteAddressBuffer material_data);

// The four built-in samplers. Each defines SampleMaterial and, for shadowing,
// SampleShadow / SampleShadowOpacity. The macros they set are scoped to their
// own namespace, as in the generated GPU source.
namespace SoftwareMaterialLambertian {
#include "material/lambertian/sampler.hlsl"
#include "sparkium/pipelines/raytracing/cpu/shaders/hlsl_cpu_transmission.inl"
}  // namespace SoftwareMaterialLambertian
#undef SAMPLE_SHADOW_ANY_HIT
#undef SAMPLE_SHADOW_NO_HITRECORD

namespace SoftwareMaterialLight {
#include "material/light/sampler.hlsl"
#include "sparkium/pipelines/raytracing/cpu/shaders/hlsl_cpu_transmission.inl"
}  // namespace SoftwareMaterialLight
#undef SAMPLE_SHADOW_ANY_HIT
#undef SAMPLE_SHADOW_NO_HITRECORD

namespace SoftwareMaterialPrincipled {
#include "material/principled/sampler.hlsl"
#include "sparkium/pipelines/raytracing/cpu/shaders/hlsl_cpu_transmission.inl"
}  // namespace SoftwareMaterialPrincipled
#undef SAMPLE_SHADOW_ANY_HIT
#undef SAMPLE_SHADOW_NO_HITRECORD

namespace SoftwareMaterialSpecular {
#include "material/specular/sampler.hlsl"
#include "sparkium/pipelines/raytracing/cpu/shaders/hlsl_cpu_transmission.inl"
}  // namespace SoftwareMaterialSpecular
#undef SAMPLE_SHADOW_ANY_HIT
#undef SAMPLE_SHADOW_NO_HITRECORD

namespace SoftwareMaterialShaderGraph {
#include "material/shader_graph/sampler.hlsl"
#include "sparkium/pipelines/raytracing/cpu/shaders/hlsl_cpu_transmission.inl"
}  // namespace SoftwareMaterialShaderGraph
#undef SAMPLE_SHADOW_ANY_HIT
#undef SAMPLE_SHADOW_NO_HITRECORD

// Shader-graph materials never call Transmission: their shadow opacity comes
// from the graph, matching the generated GPU dispatch.
float ShaderGraphTransmission(HitRecord hit_record, float3 direction) {
  InstanceMetadata metadata =
      instance_metadatas.Load<InstanceMetadata>(sizeof(InstanceMetadata) * hit_record.object_index);
  ByteAddressBuffer material_data = data_buffers[NonUniformResourceIndex(metadata.material_data_index)];
  return 1.0f - saturate(GraphShadowOpacity(EvaluateShaderGraph(hit_record, -direction, 1, RAY_TYPE_REFLECTION,
                                                                true, material_data)));
}

// The two entry points software/render.hlsl and software/shadow.hlsli call.
void SoftwareSampleMaterial(uint material, RenderContext &context, HitRecord hit) {
  switch (MaterialKernel(material_kernels[material])) {
    case MATERIAL_KERNEL_LAMBERTIAN:
      SoftwareMaterialLambertian::SampleMaterial(context, hit);
      return;
    case MATERIAL_KERNEL_LIGHT:
      SoftwareMaterialLight::SampleMaterial(context, hit);
      return;
    case MATERIAL_KERNEL_PRINCIPLED:
      SoftwareMaterialPrincipled::SampleMaterial(context, hit);
      return;
    case MATERIAL_KERNEL_SPECULAR:
      SoftwareMaterialSpecular::SampleMaterial(context, hit);
      return;
    case MATERIAL_KERNEL_SHADER_GRAPH:
      SoftwareMaterialShaderGraph::SampleMaterial(context, hit);
      return;
    default:
      context.throughput = float3(0, 0, 0);
      return;
  }
}

float SoftwareShadowTransmission(uint material, HitRecord hit, float3 direction) {
  switch (MaterialKernel(material_kernels[material])) {
    case MATERIAL_KERNEL_LAMBERTIAN:
      return SoftwareMaterialLambertian::Transmission(hit, direction);
    case MATERIAL_KERNEL_LIGHT:
      return SoftwareMaterialLight::Transmission(hit, direction);
    case MATERIAL_KERNEL_PRINCIPLED:
      return SoftwareMaterialPrincipled::Transmission(hit, direction);
    case MATERIAL_KERNEL_SPECULAR:
      return SoftwareMaterialSpecular::Transmission(hit, direction);
    case MATERIAL_KERNEL_SHADER_GRAPH:
      return ShaderGraphTransmission(hit, direction);
    default:
      return 0.0f;
  }
}

}  // namespace sparkium_cpu_shaders
