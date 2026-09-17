#pragma once
#include "sparkium/core/entity.h"

namespace sparkium {

class EntityPointLight : public Entity {
 public:
  EntityPointLight(Core *core,
                   const glm::vec3 &position = {},
                   const glm::vec3 &color = {1.0f, 1.0f, 1.0f},
                   float strength = 0.0f,
                   float radius = 0.0f,
                   bool soft_falloff = false,
                   float sampling_weight = -1.0f);

  glm::vec3 position;
  glm::vec3 color;
  float strength;
  float radius;
  int soft_falloff;
  float sampling_weight;
};

}  // namespace sparkium
