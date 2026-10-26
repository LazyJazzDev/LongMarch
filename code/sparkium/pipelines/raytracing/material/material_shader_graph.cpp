#include "sparkium/pipelines/raytracing/material/material_shader_graph.h"

#include <stdexcept>

#include "sparkium/pipelines/raytracing/core/core.h"
#include "sparkium/pipelines/raytracing/core/scene.h"

namespace sparkium::raytracing {

MaterialShaderGraph::MaterialShaderGraph(sparkium::MaterialShaderGraph &material)
    : Material(DedicatedCast(material.GetCore())),
      material_(material) {
  const size_t size = std::max<size_t>(16, 12 + material_.textures.size() * sizeof(int));
  material_buffer_ = core_->GetDevice()->CreateBuffer({size});
  sampler_implementation_ = CodeLines(core_->GetShadersVFS(), "material/shader_graph/sampler.slang");
  sampler_implementation_.InsertAfter(material_.graph_code, "// SHADER_GRAPH_IMPLEMENTATION");
  evaluator_implementation_ = CodeLines(core_->GetShadersVFS(), "material/shader_graph/evaluator.slang");
  UploadBuffer(core_->GetDevice(), material_buffer_.Get(), &material_.emission_hint, sizeof(material_.emission_hint));
}

void MaterialShaderGraph::PrepareHitShaders() {
  if (closest_hit_shader_ && shadow_closest_hit_shader_ && shadow_any_hit_shader_)
    return;
  auto vfs = core_->GetShadersVFS();
  vfs.WriteFile("material_sampler.slang", sampler_implementation_);
  if (!core_->GetDevice()->SupportRayTracingPipeline())
    return;
  closest_hit_shader_ = core_->GetDevice()->CreateShader("geometry/mesh/hit_group.slang", "RenderClosestHit",
                                                         rhi::SHADER_STAGE_CLOSEST_HIT, &vfs, {"."});
  shadow_closest_hit_shader_ = core_->GetDevice()->CreateShader("geometry/mesh/hit_group.slang", "ShadowClosestHit",
                                                                rhi::SHADER_STAGE_CLOSEST_HIT, &vfs, {"."});
  shadow_any_hit_shader_ = core_->GetDevice()->CreateShader("geometry/mesh/hit_group.slang", "ShadowAnyHit",
                                                            rhi::SHADER_STAGE_ANY_HIT, &vfs, {"."});
  if (!closest_hit_shader_ || !shadow_closest_hit_shader_ || !shadow_any_hit_shader_)
    throw std::runtime_error("failed to compile shader graph material");
}

const CodeLines *MaterialShaderGraph::GraphImpl() const {
  return &material_.graph_code;
}

rhi::IBuffer *MaterialShaderGraph::Buffer() {
  return material_buffer_.Get();
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
    UploadBuffer(core_->GetDevice(), material_buffer_.Get(), indices.data(), indices.size() * sizeof(int), 12);
}

}  // namespace sparkium::raytracing
