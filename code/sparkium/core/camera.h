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

 private:
  Core *core_;
};

}  // namespace sparkium
