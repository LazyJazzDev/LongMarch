#pragma once
#include "sparkium/core/material.h"

namespace sparkium {

class MaterialSpecular : public Material {
 public:
  MaterialSpecular(Core *core, const glm::vec3 &base_color = glm::vec3{0.8f});

  glm::vec3 base_color{0.8f};
};

}  // namespace sparkium
