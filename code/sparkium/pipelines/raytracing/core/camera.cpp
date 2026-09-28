#include "sparkium/pipelines/raytracing/core/camera.h"

#include <stdexcept>

#include "sparkium/camera/cameras.h"
#include "sparkium/pipelines/raytracing/camera/camera_pinhole.h"
#include "sparkium/pipelines/raytracing/camera/camera_thin_lens.h"
#include "sparkium/pipelines/raytracing/core/core.h"

namespace sparkium::raytracing {

Camera::Camera(Core *core, const std::string &shader_file, const std::string &entry_point)
    : core_(core),
      shader_file_(shader_file),
      entry_point_(entry_point) {
}

graphics::Shader *Camera::Shader() {
  if (!camera_shader_ && core_->GraphicsCore()->DeviceRayTracingSupport()) {
    if (core_->GraphicsCore()->CreateShader(core_->GetShadersVFS(), shader_file_, entry_point_, "lib_6_5",
                                            &camera_shader_))
      throw std::runtime_error("failed to compile camera shader");
  }
  return camera_shader_.get();
}

const std::string &Camera::ShaderFile() const {
  return shader_file_;
}

const std::string &Camera::EntryPoint() const {
  return entry_point_;
}

Camera *DedicatedCast(sparkium::Camera *camera) {
  DEDICATED_CAST(camera, sparkium::CameraPinhole, CameraPinhole);
  DEDICATED_CAST(camera, sparkium::CameraThinLens, CameraThinLens);
  throw std::invalid_argument("unsupported ray tracing camera model");
}

}  // namespace sparkium::raytracing
