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
  rhi::IBuffer *Buffer() override;

 private:
  sparkium::CameraPinhole &camera_;
  rhi::ObjPtr<rhi::IBuffer> buffer_;
};

}  // namespace sparkium::raytracing
