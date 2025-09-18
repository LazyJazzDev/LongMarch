#pragma once
#include "sparkium/core/core_util.h"

namespace sparkium {

struct CameraData {
  glm::mat4 world_to_camera;
  glm::mat4 camera_to_world;
  glm::vec2 scale;
};

class Camera {
 public:
  Camera(Core *core, const glm::mat4 &view, float fovy, float aspect);

  graphics::Shader *Shader() const {
    return camera_shader_.get();
  }

  graphics::Buffer *Buffer() const {
    return camera_buffer_.get();
  }

 private:
  Core *core_;
  std::unique_ptr<graphics::Shader> camera_shader_;
  std::unique_ptr<graphics::Buffer> camera_buffer_;
  CameraData camera_data_;
};

}  // namespace sparkium
