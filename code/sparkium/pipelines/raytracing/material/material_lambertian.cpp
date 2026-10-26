#include "sparkium/pipelines/raytracing/material/material_lambertian.h"

#include "sparkium/pipelines/raytracing/core/core.h"

namespace sparkium::raytracing {

MaterialLambertian::MaterialLambertian(sparkium::MaterialLambertian &material)
    : material_(material),
      Material(DedicatedCast(material.GetCore())) {
  material_buffer_ = core_->GetDevice()->CreateBuffer({sizeof(material_.base_color) + sizeof(material_.emission)});
  sampler_implementation_ = CodeLines(core_->GetShadersVFS(), "material/lambertian/sampler.slang");
  evaluator_implementation_ = CodeLines(core_->GetShadersVFS(), "material/lambertian/evaluator.slang");
}

rhi::IBuffer *MaterialLambertian::Buffer() {
  SyncMaterialData();
  return material_buffer_.Get();
}

const CodeLines &MaterialLambertian::SamplerImpl() const {
  return sampler_implementation_;
}

const CodeLines &MaterialLambertian::EvaluatorImpl() const {
  return evaluator_implementation_;
}

void MaterialLambertian::SyncMaterialData() {
  std::vector<uint8_t> data(material_buffer_->GetDesc().size);
  std::memcpy(data.data(), &material_.base_color, sizeof(material_.base_color));
  std::memcpy(data.data() + sizeof(material_.base_color), &material_.emission, sizeof(material_.emission));
  UploadBuffer(core_->GetDevice(), material_buffer_.Get(), data.data(), data.size());
}

}  // namespace sparkium::raytracing
