#pragma once
#include "sparkium/camera/camera_thin_lens.h"
#include "sparkium/pipelines/raytracing/core/camera.h"

namespace sparkium::raytracing {

struct CameraThinLensData {
  glm::mat4 world_to_camera;
  glm::mat4 camera_to_world;
  glm::vec2 scale;
  float aperture_radius;
  float focus_distance;
  int aperture_blades;
  float aperture_rotation;
  float aperture_ratio;
};

class CameraThinLens : public Camera {
 public:
  explicit CameraThinLens(sparkium::CameraThinLens &camera);
  graphics::Buffer *Buffer() override;

 private:
  sparkium::CameraThinLens &camera_;
  std::unique_ptr<graphics::Buffer> buffer_;
};

}  // namespace sparkium::raytracing
