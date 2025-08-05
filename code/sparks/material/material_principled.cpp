#include "sparks/material/material_principled.h"

#include "sparks/core/core.h"

namespace sparks {

MaterialPrincipled::MaterialPrincipled(Core *core, const glm::vec3 &base_color)
    : Material(core), base_color(base_color) {
  core_->GraphicsCore()->CreateBuffer(sizeof(glm::vec3) * 3 + sizeof(float) * 14, graphics::BUFFER_TYPE_STATIC,
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
  uint8_t *data_ptr = data.data();
  auto write_data = [&](const auto &value) {
    std::memcpy(data_ptr, &value, sizeof(value));
    data_ptr += sizeof(value);
  };

  // glm::vec3 base_color{0.8f};
  //
  // glm::vec3 subsurface_color{1.0f, 1.0f, 1.0f};
  // float subsurface{0.0f};
  //
  // glm::vec3 subsurface_radius{1.0f, 0.2f, 0.1f};
  // float metallic{0.05f};
  //
  // float specular{0.0f};
  // float specular_tint{0.0f};
  // float roughness{0.0f};
  // float anisotropic{0.0f};
  //
  // float anisotropic_rotation{0.0f};
  // float sheen{0.0f};
  // float sheen_tint{0.0f};
  // float clearcoat{0.0f};
  //
  // float clearcoat_roughness{0.0f};
  // float ior{1.2f};
  // float transmission{0.0f};
  // float transmission_roughness{0.0f};
  write_data(base_color);
  write_data(subsurface_color);
  write_data(subsurface);
  write_data(subsurface_radius);
  write_data(metallic);
  write_data(specular);
  write_data(specular_tint);
  write_data(roughness);
  write_data(anisotropic);
  write_data(anisotropic_rotation);
  write_data(sheen);
  write_data(sheen_tint);
  write_data(clearcoat);
  write_data(clearcoat_roughness);
  write_data(ior);
  write_data(transmission);
  write_data(transmission_roughness);

  material_buffer_->UploadData(data.data(), data.size());
}

}  // namespace sparks
