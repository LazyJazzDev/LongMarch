#pragma once
#include "sparkium/core/core_util.h"

namespace sparkium {

struct CameraData {
  glm::mat4 world_to_camera;
  glm::mat4 camera_to_world;
  glm::vec2 scale;
};

class Camera : public Object {
 public:
  Camera(Core *core, const glm::mat4 &view, float fovy, float aspect);
  glm::mat4 view;
  float fovy;
  float aspect;

 private:
  Core *core_;
};

}  // namespace sparkium
