#include "sparks/surface/surface_lambertian.h"

#include "sparks/core/core.h"

namespace sparks {

SurfaceLambertian::SurfaceLambertian(Core *core, const glm::vec3 &base_color, const glm::vec3 &emission)
    : Surface(core), base_color(base_color), emission(emission) {
  core_->GraphicsCore()->CreateShader(core_->GetShadersVFS(), "surface/lambertian/main.hlsl", "CallableMain", "lib_6_3",
                                      {"-I."}, &callable_shader_);
  core_->GraphicsCore()->CreateBuffer(sizeof(base_color) + sizeof(emission), graphics::BUFFER_TYPE_STATIC,
                                      &surface_buffer_);
  std::vector<uint8_t> sampler_impl_data;
  core_->GetShadersVFS().ReadFile("surface/lambertian/surface_sampler.hlsli", sampler_impl_data);
  sampler_implementation_ = sampler_impl_data;
  SyncSurfaceData();
}

graphics::Buffer *SurfaceLambertian::Buffer() {
  SyncSurfaceData();
  return surface_buffer_.get();
}

graphics::Shader *SurfaceLambertian::CallableShader() {
  return callable_shader_.get();
}

const CodeLines &SurfaceLambertian::SamplerImplementation() const {
  return sampler_implementation_;
}

void SurfaceLambertian::SyncSurfaceData() {
  std::vector<uint8_t> data(surface_buffer_->Size());
  std::memcpy(data.data(), &base_color, sizeof(base_color));
  std::memcpy(data.data() + sizeof(base_color), &emission, sizeof(emission));
  surface_buffer_->UploadData(data.data(), data.size());
}

}  // namespace sparks
