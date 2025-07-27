#pragma once
#include "sparks/core/core_util.h"

namespace sparks {

struct MaterialLambertian {
  glm::vec3 base_color{0.8f};
  glm::vec3 emissive_color{0.0f};
};

class Material {
 public:
  Material(Core *core, const MaterialLambertian &material = {});

  graphics::Buffer *Buffer() {
    return material_buffer_.get();
  }
  graphics::Shader *CallableShader() {
    return callable_shader_.get();
  }

  MaterialLambertian material;

 private:
  Core *core_;
  std::unique_ptr<graphics::Buffer> material_buffer_;
  std::unique_ptr<graphics::Shader> callable_shader_;
};

}  // namespace sparks
