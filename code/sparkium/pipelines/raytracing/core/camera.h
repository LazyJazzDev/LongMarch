#pragma once
#include "sparkium/pipelines/raytracing/core/core_util.h"

namespace sparkium::raytracing {

struct CameraData {
  glm::mat4 world_to_camera;
  glm::mat4 camera_to_world;
  glm::vec2 scale;
};

class Camera : public Object {
 public:
  Camera(sparkium::Camera &camera);

  graphics::Shader *Shader() const;

  graphics::Buffer *Buffer();

 private:
  sparkium::Camera &camera_;
  Core *core_;
  std::unique_ptr<graphics::Shader> camera_shader_;
  std::unique_ptr<graphics::Buffer> camera_buffer_;
  CameraData camera_data_;
};

Camera *DedicatedCast(sparkium::Camera *camera);

}  // namespace sparkium::raytracing
