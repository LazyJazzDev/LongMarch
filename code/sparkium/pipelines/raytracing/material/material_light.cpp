#include "sparkium/pipelines/raytracing/material/material_light.h"

#include "sparkium/pipelines/raytracing/core/core.h"

namespace sparkium::raytracing {

MaterialLight::MaterialLight(sparkium::MaterialLight &material)
    : material_(material), Material(DedicatedCast(material.GetCore())) {
  core_->GraphicsCore()->CreateBuffer(sizeof(material_.emission) + sizeof(int) * 2, graphics::BUFFER_TYPE_STATIC,
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
  std::memcpy(data.data(), &material_.emission, sizeof(material_.emission));
  std::memcpy(data.data() + sizeof(material_.emission), &material_.two_sided, sizeof(material_.two_sided));
  std::memcpy(data.data() + sizeof(material_.emission) + sizeof(material_.two_sided), &material_.block_ray,
              sizeof(material_.block_ray));
  material_buffer_->UploadData(data.data(), data.size());
}

}  // namespace sparkium::raytracing
