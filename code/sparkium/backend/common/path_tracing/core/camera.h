#pragma once
#include "sparkium/backend/common/path_tracing/core/core_util.h"

namespace sparkium::raytracing {

struct CameraData {
  glm::mat4 world_to_camera;
  glm::mat4 camera_to_world;
  glm::vec2 scale;
  float aperture_radius;
  float focus_distance;
  int aperture_blades;
  float aperture_rotation;
  float aperture_ratio;
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
