#pragma once
#include "sparkium/core/core_util.h"

namespace sparkium {

class Camera : public Object {
 public:
  Camera(Core *core, const glm::mat4 &view, float fovy, float aspect);
  Core *GetCore() const;

  glm::mat4 view;
  float fovy;
  float aspect;
  float aperture_radius{0.0f};
  float focus_distance{1.0f};
  int aperture_blades{0};
  float aperture_rotation{0.0f};
  float aperture_ratio{1.0f};

 private:
  Core *core_;
};

}  // namespace sparkium
