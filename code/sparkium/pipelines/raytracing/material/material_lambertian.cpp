#include "sparkium/pipelines/raytracing/material/material_lambertian.h"

#include "sparkium/pipelines/raytracing/core/core.h"

namespace sparkium::raytracing {

MaterialLambertian::MaterialLambertian(sparkium::MaterialLambertian &material)
    : material_(material), Material(DedicatedCast(material.GetCore())) {
  core_->GraphicsCore()->CreateBuffer(sizeof(material_.base_color) + sizeof(material_.emission),
                                      graphics::BUFFER_TYPE_STATIC, &material_buffer_);
  sampler_implementation_ = CodeLines(core_->GetShadersVFS(), "material/lambertian/sampler.hlsl");
  evaluator_implementation_ = CodeLines(core_->GetShadersVFS(), "material/lambertian/evaluator.hlsli");
}

graphics::Buffer *MaterialLambertian::Buffer() {
  SyncMaterialData();
  return material_buffer_.get();
}

const CodeLines &MaterialLambertian::SamplerImpl() const {
  return sampler_implementation_;
}

const CodeLines &MaterialLambertian::EvaluatorImpl() const {
  return evaluator_implementation_;
}

void MaterialLambertian::SyncMaterialData() {
  std::vector<uint8_t> data(material_buffer_->Size());
  std::memcpy(data.data(), &material_.base_color, sizeof(material_.base_color));
  std::memcpy(data.data() + sizeof(material_.base_color), &material_.emission, sizeof(material_.emission));
  material_buffer_->UploadData(data.data(), data.size());
}

}  // namespace sparkium::raytracing
