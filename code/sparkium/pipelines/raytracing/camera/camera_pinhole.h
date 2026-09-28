#pragma once
#include "sparkium/camera/camera_pinhole.h"
#include "sparkium/pipelines/raytracing/core/camera.h"

namespace sparkium::raytracing {

struct CameraPinholeData {
  glm::mat4 world_to_camera;
  glm::mat4 camera_to_world;
  glm::vec2 scale;
};

class CameraPinhole : public Camera {
 public:
  explicit CameraPinhole(sparkium::CameraPinhole &camera);
  graphics::Buffer *Buffer() override;

 private:
  sparkium::CameraPinhole &camera_;
  std::unique_ptr<graphics::Buffer> buffer_;
};

}  // namespace sparkium::raytracing
