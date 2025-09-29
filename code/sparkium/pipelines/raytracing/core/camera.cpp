#include "sparkium/pipelines/raytracing/core/camera.h"

#include "sparkium/core/camera.h"
#include "sparkium/pipelines/raytracing/core/core.h"

namespace sparkium::raytracing {

Camera::Camera(sparkium::Camera &camera) : camera_(camera) {
  core_ = DedicatedCast(camera_.GetCore());
  core_->GraphicsCore()->CreateShader(core_->GetShadersVFS(), "camera.hlsl", "CameraPinhole", "lib_6_5",
                                      &camera_shader_);
  core_->GraphicsCore()->CreateBuffer(sizeof(CameraData), graphics::BUFFER_TYPE_STATIC, &camera_buffer_);
  camera_data_.world_to_camera = camera_.view;
  camera_data_.camera_to_world = glm::inverse(camera_.view);
  camera_data_.scale = glm::vec2(camera_.aspect * tan(camera_.fovy * 0.5f), tan(camera_.fovy * 0.5f));
  camera_buffer_->UploadData(&camera_data_, sizeof(camera_data_));
}

graphics::Shader *Camera::Shader() const {
  return camera_shader_.get();
}

graphics::Buffer *Camera::Buffer() const {
  return camera_buffer_.get();
}

Camera *DedicatedCast(sparkium::Camera *camera) {
  COMPONENT_CAST(camera, Camera);
}

}  // namespace sparkium::raytracing
