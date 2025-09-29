#include "sparkium/pipelines/raytracing/material/material_specular.h"

#include "sparkium/pipelines/raytracing/core/core.h"
#include "sparkium/pipelines/raytracing/core/scene.h"

namespace sparkium::raytracing {

MaterialSpecular::MaterialSpecular(sparkium::MaterialSpecular &material)
    : material_(material), Material(DedicatedCast(material.GetCore())) {
  core_->GraphicsCore()->CreateBuffer(sizeof(material_.base_color), graphics::BUFFER_TYPE_STATIC, &material_buffer_);
  sampler_implementation_ = CodeLines(core_->GetShadersVFS(), "material/specular/sampler.hlsl");
  evaluator_implementation_ = CodeLines(core_->GetShadersVFS(), "material/specular/evaluator.hlsli");
}

graphics::Buffer *MaterialSpecular::Buffer() {
  SyncMaterialData();
  return material_buffer_.get();
}

const CodeLines &MaterialSpecular::SamplerImpl() const {
  return sampler_implementation_;
}

const CodeLines &MaterialSpecular::EvaluatorImpl() const {
  return evaluator_implementation_;
}

void MaterialSpecular::SyncMaterialData() {
  std::vector<uint8_t> data(material_buffer_->Size());
  std::memcpy(data.data(), &material_.base_color, sizeof(material_.base_color));
  material_buffer_->UploadData(data.data(), data.size());
}

}  // namespace sparkium::raytracing
