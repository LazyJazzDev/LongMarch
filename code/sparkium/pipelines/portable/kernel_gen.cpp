#include "sparkium/pipelines/portable/kernel_gen.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

#include "sparkium/core/core.h"
#include "sparkium/pipelines/portable/transpile.h"

namespace sparkium::portable {

std::string GenerateKernelSource(sparkium::Core *core, const std::vector<MaterialData> &materials) {
  Transpiler transpiler(core->GetShadersVFS());
  const bool has_graph =
      std::any_of(materials.begin(), materials.end(), [](const MaterialData &m) { return m.shader_graph; });

  std::ostringstream out;
  // Shared HLSL shading sources, transpiled once. The order mirrors
  // shaders/software/render.hlsl. Include guards make the expansion
  // idempotent, so only the leaf kernels are listed here.
  std::map<std::string, std::string> defines;
  defines["SPARKIUM_SOFTWARE_RT"] = "1";
  defines["SOFTWARE_EXTERNAL_BINDINGS"] = "1";
  if (std::getenv("SPARKIUM_PORTABLE_DEBUG_DUMP"))
    defines["SPARKIUM_PORTABLE_DEBUG"] = "1";
  static const char *kSharedFiles[] = {
      "common.hlsli",
      "buffer_helper.hlsli",
      "bsdf/principled_material.hlsli",
      "bsdf/lambertian.hlsli",
      "bsdf/specular.hlsli",
      "camera.hlsl",
      "direct_lighting.hlsli",
      "geometry/mesh/hit_record.hlsli",
      "software/traversal.hlsli",
      "subsurface_random_walk.hlsli",
  };
  for (const char *file : kSharedFiles)
    out << transpiler.Process(file, defines) << '\n';

  if (has_graph)
    out << transpiler.Process("material/shader_graph/surface_sampler.hlsli", defines) << '\n';

  // Material sampler bodies, namespaced exactly like the compute renderer's
  // generated software_materials.hlsli.
  for (size_t i = 0; i < materials.size(); ++i) {
    out << "namespace SoftwareMaterial" << i << " {\n";
    out << transpiler.TransformSnippet(materials[i].source);
    if (materials[i].shader_graph) {
      out << "}\n";
      continue;
    }
    // Mirror shaders/software/render.hlsl Transmission dispatch. The
    // SAMPLE_SHADOW_* markers were recorded while transpiling the material
    // snippet (the compute renderer keeps them as #defines).
    out << "SPARKIUM_KERNEL float Transmission(HitRecord hit, sparkium_portable::float3 direction) {\n";
    if (materials[i].shadow_any_hit) {
      out << "  return 1.0f - vsaturate(SampleShadowOpacity(hit, direction));\n";
    } else {
      out << "  ShadowRayPayload payload;\n  payload.shadow = 1.0f;\n";
      if (materials[i].shadow_no_hitrecord)
        out << "  SampleShadow(payload);\n";
      else
        out << "  SampleShadow(payload, hit);\n";
      out << "  return payload.shadow;\n";
    }
    out << "}\n}\n";
  }

  // Dispatch (mirrors the software renderer's generated switch).
  out << "SPARKIUM_KERNEL HitRecord SoftwareHitRecord(SoftwareHit hit, sparkium_portable::float3 direction) {\n"
         "  SoftwareInstance instance = LoadSoftwareInstance(ResourceSoftwareInstances(), hit.instance);\n"
         "  return MakeMeshHitRecord(instance.geometry, hit.instance, hit.primitive, hit.barycentric, "
         "hit.distance, direction, instance.object_to_world, transpose(instance.world_to_object));\n"
         "}\n";
  if (has_graph) {
    out << R"(
ByteAddressBuffer SoftwareMaterialData(HitRecord hit) {
  InstanceMetadata metadata = ResourceInstanceMetadatas().Load<InstanceMetadata>(sizeof(InstanceMetadata) * hit.object_index);
  return ResourceDataBuffer()[metadata.material_data_index];
}
SPARKIUM_KERNEL void SoftwareSampleMaterial(uint material, RenderContext &context, HitRecord hit) {
  GraphSurface graph;
  switch (material) {
)";
    for (size_t i = 0; i < materials.size(); ++i) {
      out << "case " << i << ": ";
      if (materials[i].shader_graph)
        out << "graph = SoftwareMaterial" << i
            << "::EvaluateShaderGraph(hit, -context.direction, context.bounce, context.ray_type, false, "
               "SoftwareMaterialData(hit)); break;\n";
      else
        out << "SoftwareMaterial" << i << "::SampleMaterial(context, hit); return;\n";
    }
    out << R"(
    default: context.throughput = sparkium_portable::float3(0, 0, 0); return;
  }
  SampleGraphSurface(context, hit, graph);
}
SPARKIUM_KERNEL float SoftwareShadowTransmission(uint material, HitRecord hit, sparkium_portable::float3 direction) {
  switch (material) {
)";
    for (size_t i = 0; i < materials.size(); ++i) {
      out << "case " << i << ": return ";
      if (materials[i].shader_graph)
        out << "1.0f - vsaturate(GraphShadowOpacity(SoftwareMaterial" << i
            << "::EvaluateShaderGraph(hit, -direction, 1, RAY_TYPE_REFLECTION, true, SoftwareMaterialData(hit))));\n";
      else
        out << "SoftwareMaterial" << i << "::Transmission(hit, direction);\n";
    }
    out << "default: return 0.0f;\n}}\n";
  } else {
    out << "SPARKIUM_KERNEL void SoftwareSampleMaterial(uint material, RenderContext &context, HitRecord hit) {\n"
           "switch (material) {\n";
    for (size_t i = 0; i < materials.size(); ++i)
      out << "case " << i << ": SoftwareMaterial" << i << "::SampleMaterial(context, hit); return;\n";
    out << "default: context.throughput = sparkium_portable::float3(0, 0, 0); return;\n}}\n"
           "SPARKIUM_KERNEL float SoftwareShadowTransmission(uint material, HitRecord hit, sparkium_portable::float3 direction) {\nswitch (material) {\n";
    for (size_t i = 0; i < materials.size(); ++i)
      out << "case " << i << ": return SoftwareMaterial" << i << "::Transmission(hit, direction);\n";
    out << "default: return 0.0f;\n}}\n";
  }

  // Path tracing entry shared with the GPU software renderer
  // (shaders/software/render.hlsl).
  out << R"(
SPARKIUM_KERNEL void ApplyPathMiss(RenderContext &context);
SPARKIUM_KERNEL void SoftwareTracePath(RayDesc ray, RenderContext &context) {
  SoftwareHit hit;
  if (!InlineIntersect(ray, false, hit)) {
    ApplyPathMiss(context);
    return;
  }
  HitRecord record = SoftwareHitRecord(hit, ray.Direction);
  if (!ContinueSubsurfaceRandomWalk(context, record))
    SoftwareSampleMaterial(LoadSoftwareInstance(ResourceSoftwareInstances(), hit.instance).material, context,
                           record);
}
)";
  out << transpiler.Process("raygen.hlsl", defines);
  out << transpiler.Process("software/shadow.hlsli", defines);

  // Portable entry point.
  out << R"(
#ifdef __CUDACC__
extern "C" __global__ void PortableRenderKernel() {
  const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < sparkium_portable::g_ctx.image_width && y < sparkium_portable::g_ctx.image_height)
    RenderPixel(sparkium_portable::uint2(x, y), sparkium_portable::uint2(sparkium_portable::g_ctx.image_width, sparkium_portable::g_ctx.image_height));
}
#else
extern "C" void PortableRenderPixel(uint32_t pixel_x, uint32_t pixel_y) {
  const sparkium_portable::KernelContext *c = SPARKIUM_CTX;
  RenderPixel(sparkium_portable::uint2(pixel_x, pixel_y), sparkium_portable::uint2(c->image_width, c->image_height));
}
#endif
)";
  return out.str();
}

}  // namespace sparkium::portable
