#include "sparkium/core/camera.h"

#include "core.h"

namespace sparkium {
Camera::Camera(Core *core, const glm::mat4 &view, float fovy, float aspect) : core_(core) {
  core_->GraphicsCore()->CreateShader(core_->GetShadersVFS(), "camera.hlsl", "CameraPinhole", "lib_6_5",
                                      &camera_shader_);
  core_->GraphicsCore()->CreateBuffer(sizeof(CameraData), graphics::BUFFER_TYPE_STATIC, &camera_buffer_);
  camera_data_.world_to_camera = view;
  camera_data_.camera_to_world = glm::inverse(view);
  camera_data_.scale = glm::vec2(aspect * tan(fovy * 0.5f), tan(fovy * 0.5f));
  camera_buffer_->UploadData(&camera_data_, sizeof(camera_data_));
}
}  // namespace sparkium
