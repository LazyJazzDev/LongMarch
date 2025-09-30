#pragma once
#include "sparkium/core/material.h"

namespace sparkium {

class MaterialLight : public Material {
 public:
  MaterialLight(Core *core,
                const glm::vec3 &emission = glm::vec3{0.0f},
                bool two_sided = false,
                bool block_ray = false);

  glm::vec3 emission{0.0f};
  int two_sided{0};
  int block_ray{0};
};

}  // namespace sparkium
