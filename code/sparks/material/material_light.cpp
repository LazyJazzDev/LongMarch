#include "sparks/material/material_light.h"

#include "sparks/core/core.h"

namespace XH {

MaterialLight::MaterialLight(Core *core, const glm::vec3 &emission, bool two_sided, bool block_ray)
    : Material(core), emission(emission), two_sided(two_sided), block_ray(block_ray) {
  core_->GraphicsCore()->CreateBuffer(sizeof(emission) + sizeof(int) * 2, graphics::BUFFER_TYPE_STATIC,
                                      &material_buffer_);
  sampler_implementation_ = CodeLines(core_->GetShadersVFS(), "material/light/sampler.hlsl");
  evaluator_implementation_ = CodeLines(core_->GetShadersVFS(), "material/light/evaluator.hlsli");
  SyncMaterialData();
}

graphics::Buffer *MaterialLight::Buffer() {
  SyncMaterialData();
  return material_buffer_.get();
}

const CodeLines &MaterialLight::SamplerImpl() const {
  return sampler_implementation_;
}

const CodeLines &MaterialLight::EvaluatorImpl() const {
  return evaluator_implementation_;
}

void MaterialLight::SyncMaterialData() {
  std::vector<uint8_t> data(material_buffer_->Size());
  std::memcpy(data.data(), &emission, sizeof(emission));
  std::memcpy(data.data() + sizeof(emission), &two_sided, sizeof(two_sided));
  std::memcpy(data.data() + sizeof(emission) + sizeof(two_sided), &block_ray, sizeof(block_ray));
  material_buffer_->UploadData(data.data(), data.size());
}

}  // namespace XH
