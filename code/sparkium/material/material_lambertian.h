#pragma once
#include "sparkium/core/material.h"

namespace sparkium {

class MaterialLambertian : public Material {
 public:
  MaterialLambertian(Core *core,
                     const glm::vec3 &base_color = glm::vec3{0.8f},
                     const glm::vec3 &emission = glm::vec3{0.0f});

  glm::vec3 base_color{0.8f};
  glm::vec3 emission{0.0f};
};

}  // namespace sparkium
