#include "sparks/material/material_principled.h"

#include "sparks/core/core.h"

namespace sparks {

MaterialPrincipled::MaterialPrincipled(Core *core, const glm::vec3 &base_color)
    : Material(core), base_color(base_color) {
  core_->GraphicsCore()->CreateBuffer(sizeof(glm::vec3) * 4 + sizeof(float) * 15, graphics::BUFFER_TYPE_STATIC,
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

  write_data(base_color);              // offset 0
  write_data(subsurface_color);        // offset 12
  write_data(subsurface);              // offset 24
  write_data(subsurface_radius);       // offset 28
  write_data(metallic);                // offset 40
  write_data(specular);                // offset 44
  write_data(specular_tint);           // offset 48
  write_data(roughness);               // offset 52
  write_data(anisotropic);             // offset 56
  write_data(anisotropic_rotation);    // offset 60
  write_data(sheen);                   // offset 64
  write_data(sheen_tint);              // offset 68
  write_data(clearcoat);               // offset 72
  write_data(clearcoat_roughness);     // offset 76
  write_data(ior);                     // offset 80
  write_data(transmission);            // offset 84
  write_data(transmission_roughness);  // offset 88
  write_data(emission_color);          // offset 92
  write_data(emission_strength);       // offset 104

  material_buffer_->UploadData(data.data(), data.size());
}

}  // namespace sparks
