#pragma once
#include "sparkium/core/entity.h"

namespace sparkium {

class EntityPointLight : public Entity {
 public:
  EntityPointLight(Core *core,
                   const glm::vec3 &position = {},
                   const glm::vec3 &color = {1.0f, 1.0f, 1.0f},
                   float strength = 0.0f);

  glm::vec3 position;
  glm::vec3 color;
  float strength;
};

}  // namespace sparkium
