#pragma once
#include "sparkium/core/camera.h"

namespace sparkium {

class CameraPinhole : public Camera {
 public:
  CameraPinhole(Core *core, const glm::mat4 &view, float fovy, float aspect);
};

}  // namespace sparkium
