#include "sparks/material/material_principled.h"

#include "sparks/core/core.h"

namespace sparks {

MaterialPrincipled::MaterialPrincipled(Core *core, const glm::vec3 &base_color, const glm::vec3 &emission)
    : Material(core), base_color(base_color), emission(emission) {
  core_->GraphicsCore()->CreateBuffer(sizeof(base_color) + sizeof(emission), graphics::BUFFER_TYPE_STATIC,
                                      &material_buffer_);
  sampler_implementation_ = CodeLines(core_->GetShadersVFS(), "material/principled/sampler.hlsl");
  evaluator_implementation_ = CodeLines(core_->GetShadersVFS(), "material/principled/evaluator.hlsli");
}

graphics::Buffer *MaterialPrincipled::Buffer() {
  SyncMaterialData();
  return material_buffer_.get();
}

const CodeLines &MaterialPrincipled::SamplerImpl() const {
  return sampler_implementation_;
}

const CodeLines &MaterialPrincipled::EvaluatorImpl() const {
  return evaluator_implementation_;
}

void MaterialPrincipled::SyncMaterialData() {
  std::vector<uint8_t> data(material_buffer_->Size());
  std::memcpy(data.data(), &base_color, sizeof(base_color));
  std::memcpy(data.data() + sizeof(base_color), &emission, sizeof(emission));
  material_buffer_->UploadData(data.data(), data.size());
}

}  // namespace sparks
