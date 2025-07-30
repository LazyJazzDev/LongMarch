#include "sparks/material/material_lambertian.h"

#include "sparks/core/core.h"

namespace sparks {

MaterialLambertian::MaterialLambertian(Core *core, const glm::vec3 &base_color, const glm::vec3 &emission)
    : Material(core), base_color(base_color), emission(emission) {
  core_->GraphicsCore()->CreateShader(core_->GetShadersVFS(), "material/lambertian/main.hlsl", "CallableMain",
                                      "lib_6_3", {"-I."}, &callable_shader_);
  core_->GraphicsCore()->CreateBuffer(sizeof(base_color) + sizeof(emission), graphics::BUFFER_TYPE_STATIC,
                                      &material_buffer_);
}
graphics::Buffer *MaterialLambertian::Buffer() {
  std::vector<uint8_t> data(material_buffer_->Size());
  std::memcpy(data.data(), &base_color, sizeof(base_color));
  std::memcpy(data.data() + sizeof(base_color), &emission, sizeof(emission));
  material_buffer_->UploadData(data.data(), data.size());
  return material_buffer_.get();
}

graphics::Shader *MaterialLambertian::CallableShader() {
  return callable_shader_.get();
}

}  // namespace sparks
