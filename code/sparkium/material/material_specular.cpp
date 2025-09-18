#include "sparkium/material/material_specular.h"

#include "sparkium/core/core.h"

namespace sparkium {

MaterialSpecular::MaterialSpecular(Core *core, const glm::vec3 &base_color) : Material(core), base_color(base_color) {
  core_->GraphicsCore()->CreateBuffer(sizeof(base_color), graphics::BUFFER_TYPE_STATIC, &material_buffer_);
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
  std::memcpy(data.data(), &base_color, sizeof(base_color));
  material_buffer_->UploadData(data.data(), data.size());
}

}  // namespace sparkium
