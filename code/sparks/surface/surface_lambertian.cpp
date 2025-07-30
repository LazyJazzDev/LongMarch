#include "sparks/surface/surface_lambertian.h"

#include "sparks/core/core.h"

namespace sparks {

SurfaceLambertian::SurfaceLambertian(Core *core, const glm::vec3 &base_color, const glm::vec3 &emission)
    : Surface(core), base_color(base_color), emission(emission) {
  core_->GraphicsCore()->CreateShader(core_->GetShadersVFS(), "surface/lambertian/main.hlsl", "CallableMain", "lib_6_3",
                                      {"-I."}, &callable_shader_);
  core_->GraphicsCore()->CreateBuffer(sizeof(base_color) + sizeof(emission), graphics::BUFFER_TYPE_STATIC,
                                      &surface_buffer_);
}
graphics::Buffer *SurfaceLambertian::Buffer() {
  std::vector<uint8_t> data(surface_buffer_->Size());
  std::memcpy(data.data(), &base_color, sizeof(base_color));
  std::memcpy(data.data() + sizeof(base_color), &emission, sizeof(emission));
  surface_buffer_->UploadData(data.data(), data.size());
  return surface_buffer_.get();
}

graphics::Shader *SurfaceLambertian::CallableShader() {
  return callable_shader_.get();
}

}  // namespace sparks
