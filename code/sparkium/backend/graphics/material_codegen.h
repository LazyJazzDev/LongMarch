#pragma once
#include <algorithm>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace sparkium::backend::graphics_backend {
struct MaterialCode {
  bool shader_graph;
  std::string source;

  bool operator==(const MaterialCode &other) const {
    return shader_graph == other.shader_graph && source == other.source;
  }
};

inline std::pair<std::string, bool> GenerateMaterialDispatch(const std::vector<MaterialCode> &materials) {
  const bool has_graph = std::any_of(materials.begin(), materials.end(),
                                     [](const MaterialCode &material) { return material.shader_graph; });
  std::ostringstream source;
  for (size_t i = 0; i < materials.size(); ++i) {
    source << "namespace SoftwareMaterial" << i << " {\n" << materials[i].source;
    if (materials[i].shader_graph) {
      source << "}\n";
      continue;
    }
    source << R"(
float Transmission(SP_CONTEXT HitRecord hit, float3 direction) {
#ifdef SAMPLE_SHADOW_ANY_HIT
  return 1.0f - saturate(SampleShadowOpacity(SP_CONTEXT_ARG hit, direction));
#else
  ShadowRayPayload payload;
  payload.shadow = 1.0f;
#ifdef SAMPLE_SHADOW_NO_HITRECORD
  SampleShadow(payload);
#else
  SampleShadow(payload, hit);
#endif
  return payload.shadow;
#endif
}
}
#undef SAMPLE_SHADOW_ANY_HIT
#undef SAMPLE_SHADOW_NO_HITRECORD
)";
  }
  if (has_graph) {
    source << R"(
  ByteAddressBuffer SoftwareMaterialData(SP_CONTEXT HitRecord hit) {
    InstanceMetadata metadata = SP_BINDING_instance_metadatas.Load<InstanceMetadata>(sizeof(InstanceMetadata) * hit.object_index);
    return SP_BINDING_data_buffers[SP_NONUNIFORM(metadata.material_data_index)];
  }
  void SoftwareSampleMaterial(SP_CONTEXT uint material, inout RenderContext context, HitRecord hit) {
    GraphSurface graph;
    switch (material) {
  )";
    for (size_t i = 0; i < materials.size(); ++i) {
      source << "case " << i << ": ";
      if (materials[i].shader_graph)
        source
            << "graph = SoftwareMaterial" << i
            << "::EvaluateShaderGraph(SP_CONTEXT_ARG hit, -context.direction, context.bounce, context.ray_type, false, "
               "SoftwareMaterialData(SP_CONTEXT_ARG hit)); break;\n";
      else
        source << "SoftwareMaterial" << i << "::SampleMaterial(SP_CONTEXT_ARG context, hit); return;\n";
    }
    source << R"(
      default: context.throughput = float3(0, 0, 0); return;
    }
    SampleGraphSurface(SP_CONTEXT_ARG context, hit, graph);
  }
  float SoftwareShadowTransmission(SP_CONTEXT uint material, HitRecord hit, float3 direction) {
    switch (material) {
  )";
    for (size_t i = 0; i < materials.size(); ++i) {
      source << "case " << i << ": return ";
      if (materials[i].shader_graph)
        source << "1.0f - saturate(GraphShadowOpacity(SoftwareMaterial" << i
               << "::EvaluateShaderGraph(SP_CONTEXT_ARG hit, -direction, 1, RAY_TYPE_REFLECTION, true, "
                  "SoftwareMaterialData(SP_CONTEXT_ARG hit))));\n";
      else
        source << "SoftwareMaterial" << i << "::Transmission(SP_CONTEXT_ARG hit, direction);\n";
    }
    source << "default: return 0.0f;\n}}\n";
  } else {
    // Preserve the compact dispatch for scenes without material graphs.
    source << "void SoftwareSampleMaterial(SP_CONTEXT uint material, inout RenderContext context, HitRecord hit) {\n"
              "switch (material) {\n";
    for (size_t i = 0; i < materials.size(); ++i)
      source << "case " << i << ": SoftwareMaterial" << i << "::SampleMaterial(SP_CONTEXT_ARG context, hit); return;\n";
    source << "default: context.throughput = float3(0, 0, 0); break;\n}}\n"
              "float SoftwareShadowTransmission(SP_CONTEXT uint material, HitRecord hit, float3 direction) {\nswitch "
              "(material) {\n";
    for (size_t i = 0; i < materials.size(); ++i)
      source << "case " << i << ": return SoftwareMaterial" << i << "::Transmission(SP_CONTEXT_ARG hit, direction);\n";
    source << "default: return 0.0f;\n}}\n";
  }

  return {source.str(), has_graph};
}
}  // namespace sparkium::backend::graphics_backend
