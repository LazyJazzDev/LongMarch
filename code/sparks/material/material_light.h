#pragma once
#include "sparks/core/material.h"

namespace sparks {

class MaterialLight : public Material {
 public:
  MaterialLight(Core *core,
                const glm::vec3 &emission = glm::vec3{0.0f},
                bool two_sided = false,
                bool block_ray = false);

  graphics::Buffer *Buffer() override;
  graphics::Shader *CallableShader() override;

  glm::vec3 emission{0.0f};
  int two_sided = 0;
  int block_ray = 0;

 private:
  std::unique_ptr<graphics::Buffer> material_buffer_;
  std::unique_ptr<graphics::Shader> callable_shader_;
};

}  // namespace sparks
