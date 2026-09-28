#pragma once
#include "sparkium/core/camera.h"

namespace sparkium {

class CameraThinLens : public Camera {
 public:
  CameraThinLens(Core *core, const glm::mat4 &view, float fovy, float aspect);

  float aperture_radius{0.0f};
  float focus_distance{1.0f};
  int aperture_blades{0};
  float aperture_rotation{0.0f};
  float aperture_ratio{1.0f};
};

}  // namespace sparkium
