#include "sparkium/pipelines/raytracing/material/material_shader_graph.h"

#include <stdexcept>

#include "sparkium/pipelines/raytracing/core/core.h"
#include "sparkium/pipelines/raytracing/core/scene.h"

namespace sparkium::raytracing {

MaterialShaderGraph::MaterialShaderGraph(sparkium::MaterialShaderGraph &material)
    : Material(DedicatedCast(material.GetCore())), material_(material) {
  const size_t size = std::max<size_t>(16, 12 + material_.textures.size() * sizeof(int));
  core_->GraphicsCore()->CreateBuffer(size, graphics::BUFFER_TYPE_STATIC, &material_buffer_);
  sampler_implementation_ = CodeLines(core_->GetShadersVFS(), "material/shader_graph/sampler.hlsl");
  sampler_implementation_.InsertAfter(material_.graph_code, "// SHADER_GRAPH_IMPLEMENTATION");
  evaluator_implementation_ = CodeLines(core_->GetShadersVFS(), "material/shader_graph/evaluator.hlsli");
  auto vfs = core_->GetShadersVFS();
  vfs.WriteFile("material_sampler.hlsli", sampler_implementation_);
  if (core_->GraphicsCore()->CreateShader(vfs, "geometry/mesh/hit_group.hlsl", "RenderClosestHit", "lib_6_5",
                                          {"-I."}, &closest_hit_shader_) != 0 ||
      core_->GraphicsCore()->CreateShader(vfs, "geometry/mesh/hit_group.hlsl", "ShadowClosestHit", "lib_6_5",
                                          {"-I."}, &shadow_closest_hit_shader_) != 0 ||
      core_->GraphicsCore()->CreateShader(vfs, "geometry/mesh/hit_group.hlsl", "ShadowAnyHit", "lib_6_5",
                                          {"-I."}, &shadow_any_hit_shader_) != 0)
    throw std::runtime_error("failed to compile shader graph material");
  material_buffer_->UploadData(&material_.emission_hint, sizeof(material_.emission_hint));
}

graphics::Buffer *MaterialShaderGraph::Buffer() { return material_buffer_.get(); }
const CodeLines &MaterialShaderGraph::SamplerImpl() const { return sampler_implementation_; }
const CodeLines &MaterialShaderGraph::EvaluatorImpl() const { return evaluator_implementation_; }

void MaterialShaderGraph::Update(Scene *scene) {
  std::vector<int> indices;
  indices.reserve(material_.textures.size());
  for (auto *texture : material_.textures) indices.push_back(scene->RegisterImage(texture));
  if (!indices.empty()) material_buffer_->UploadData(indices.data(), indices.size() * sizeof(int), 12);
}

}  // namespace sparkium::raytracing
