#pragma once
#include "sparks/core/core_util.h"

namespace sparks {

struct CameraData {
  glm::mat4 world_to_camera;
  glm::mat4 camera_to_world;
  glm::vec2 scale;
};

class Camera {
 public:
  Camera(Core *core);

  graphics::Buffer *CameraDataBuffer() const;

 private:
  Core *core_;
  std::unique_ptr<graphics::Buffer> camera_buffer_;
};

}  // namespace sparks
