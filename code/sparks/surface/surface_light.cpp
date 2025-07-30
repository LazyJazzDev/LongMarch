#include "sparks/surface/surface_light.h"

#include "sparks/core/core.h"

namespace sparks {

SurfaceLight::SurfaceLight(Core *core, const glm::vec3 &emission, bool two_sided, bool block_ray)
    : Surface(core), emission(emission), two_sided(two_sided), block_ray(block_ray) {
  core_->GraphicsCore()->CreateShader(core_->GetShadersVFS(), "surface/light/main.hlsl", "CallableMain", "lib_6_3",
                                      {"-I."}, &callable_shader_);
  core_->GraphicsCore()->CreateBuffer(sizeof(emission) + sizeof(int) * 2, graphics::BUFFER_TYPE_STATIC,
                                      &surface_buffer_);
}

graphics::Buffer *SurfaceLight::Buffer() {
  std::vector<uint8_t> data(surface_buffer_->Size());
  std::memcpy(data.data(), &emission, sizeof(emission));
  std::memcpy(data.data() + sizeof(emission), &two_sided, sizeof(two_sided));
  std::memcpy(data.data() + sizeof(emission) + sizeof(two_sided), &block_ray, sizeof(block_ray));
  surface_buffer_->UploadData(data.data(), data.size());
  return surface_buffer_.get();
}

graphics::Shader *SurfaceLight::CallableShader() {
  return callable_shader_.get();
}

}  // namespace sparks
