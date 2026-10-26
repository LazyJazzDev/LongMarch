#include "sparkium/pipelines/raytracing/material/material_light.h"

#include "sparkium/pipelines/raytracing/core/core.h"

namespace sparkium::raytracing {

MaterialLight::MaterialLight(sparkium::MaterialLight &material)
    : material_(material),
      Material(DedicatedCast(material.GetCore())) {
  material_buffer_ = core_->GetDevice()->CreateBuffer({sizeof(material_.emission) + sizeof(int) * 3 + sizeof(float)});
  sampler_implementation_ = CodeLines(core_->GetShadersVFS(), "material/light/sampler.slang");
  evaluator_implementation_ = CodeLines(core_->GetShadersVFS(), "material/light/evaluator.slang");
  SyncMaterialData();
}

rhi::IBuffer *MaterialLight::Buffer() {
  SyncMaterialData();
  return material_buffer_.Get();
}

const CodeLines &MaterialLight::SamplerImpl() const {
  return sampler_implementation_;
}

const CodeLines &MaterialLight::EvaluatorImpl() const {
  return evaluator_implementation_;
}

void MaterialLight::SyncMaterialData() {
  std::vector<uint8_t> data(material_buffer_->GetDesc().size);
  std::memcpy(data.data(), &material_.emission, sizeof(material_.emission));
  std::memcpy(data.data() + sizeof(material_.emission), &material_.two_sided, sizeof(material_.two_sided));
  std::memcpy(data.data() + sizeof(material_.emission) + sizeof(material_.two_sided), &material_.block_ray,
              sizeof(material_.block_ray));
  std::memcpy(data.data() + sizeof(material_.emission) + sizeof(int) * 2, &material_.camera_visible,
              sizeof(material_.camera_visible));
  std::memcpy(data.data() + sizeof(material_.emission) + sizeof(int) * 3, &material_.falloff_distance,
              sizeof(material_.falloff_distance));
  UploadBuffer(core_->GetDevice(), material_buffer_.Get(), data.data(), data.size());
}

}  // namespace sparkium::raytracing
