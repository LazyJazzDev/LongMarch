#include "sparkium/backend/cpu/path_tracing/material/material_shader_graph.h"

#include <stdexcept>

#include "sparkium/backend/cpu/path_tracing/core/core.h"
#include "sparkium/backend/cpu/path_tracing/core/scene.h"

namespace sparkium::cpu_tracing {

MaterialShaderGraph::MaterialShaderGraph(sparkium::MaterialShaderGraph &material)
    : Material(DedicatedCast(material.GetCore())),
      material_(material) {
  const size_t size = std::max<size_t>(16, 12 + material_.textures.size() * sizeof(int));
  core_->BackendDevice()->CreateBuffer(size, graphics::BUFFER_TYPE_STATIC, &material_buffer_);
  sampler_implementation_ = CodeLines(core_->GetShadersVFS(), "material/shader_graph/sampler.hlsl");
  sampler_implementation_.InsertAfter(material_.graph_code, "// SHADER_GRAPH_IMPLEMENTATION");
  evaluator_implementation_ = CodeLines(core_->GetShadersVFS(), "material/shader_graph/evaluator.hlsli");
  material_buffer_->UploadData(&material_.emission_hint, sizeof(material_.emission_hint));
}

const CodeLines *MaterialShaderGraph::GraphImpl() const {
  return &material_.graph_code;
}

graphics::Buffer *MaterialShaderGraph::Buffer() {
  return material_buffer_.get();
}

const CodeLines &MaterialShaderGraph::SamplerImpl() const {
  return sampler_implementation_;
}

const CodeLines &MaterialShaderGraph::EvaluatorImpl() const {
  return evaluator_implementation_;
}

void MaterialShaderGraph::Update(Scene *scene) {
  std::vector<int> indices;
  indices.reserve(material_.textures.size());
  for (auto *texture : material_.textures)
    indices.push_back(scene->RegisterImage(texture));
  if (!indices.empty())
    material_buffer_->UploadData(indices.data(), indices.size() * sizeof(int), 12);
}

}  // namespace sparkium::cpu_tracing
