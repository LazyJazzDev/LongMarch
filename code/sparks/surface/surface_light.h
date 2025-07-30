#pragma once
#include "sparks/core/surface.h"

namespace sparks {

class SurfaceLight : public Surface {
 public:
  SurfaceLight(Core *core, const glm::vec3 &emission = glm::vec3{0.0f}, bool two_sided = false, bool block_ray = false);

  graphics::Buffer *Buffer() override;
  graphics::Shader *CallableShader() override;

  glm::vec3 emission{0.0f};
  int two_sided = 0;
  int block_ray = 0;

 private:
  std::unique_ptr<graphics::Buffer> surface_buffer_;
  std::unique_ptr<graphics::Shader> callable_shader_;
};

}  // namespace sparks
