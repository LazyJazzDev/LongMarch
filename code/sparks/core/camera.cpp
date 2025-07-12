#include "sparks/core/camera.h"

#include "core.h"

namespace sparks {
Camera::Camera(Core *core) : core_(core) {
  core_->GraphicsCore()->CreateBuffer(sizeof(CameraData), graphics::BUFFER_TYPE_STATIC, &camera_buffer_);
}

graphics::Buffer *Camera::CameraDataBuffer() const {
  return camera_buffer_.get();
}

}  // namespace sparks
