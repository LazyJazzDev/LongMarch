#include "sparkium/pipelines/raytracing/material/material_specular.h"

#include "sparkium/pipelines/raytracing/core/core.h"
#include "sparkium/pipelines/raytracing/core/scene.h"

namespace sparkium::raytracing {

MaterialSpecular::MaterialSpecular(sparkium::MaterialSpecular &material)
    : material_(material),
      Material(DedicatedCast(material.GetCore())) {
  material_buffer_ = core_->GetDevice()->CreateBuffer({sizeof(material_.base_color)});
  sampler_implementation_ = CodeLines(core_->GetShadersVFS(), "material/specular/sampler.slang");
  evaluator_implementation_ = CodeLines(core_->GetShadersVFS(), "material/specular/evaluator.slang");
}

rhi::IBuffer *MaterialSpecular::Buffer() {
  SyncMaterialData();
  return material_buffer_.Get();
}

const CodeLines &MaterialSpecular::SamplerImpl() const {
  return sampler_implementation_;
}

const CodeLines &MaterialSpecular::EvaluatorImpl() const {
  return evaluator_implementation_;
}

void MaterialSpecular::SyncMaterialData() {
  std::vector<uint8_t> data(material_buffer_->GetDesc().size);
  std::memcpy(data.data(), &material_.base_color, sizeof(material_.base_color));
  UploadBuffer(core_->GetDevice(), material_buffer_.Get(), data.data(), data.size());
}

}  // namespace sparkium::raytracing
