#pragma once
#include "sparkium/core/entity.h"

namespace sparkium {

class EntityGeometryLight : public Entity {
 public:
  EntityGeometryLight(Core *core,
                      Geometry *geometry,
                      const glm::vec3 &emission,
                      bool two_sided = false,
                      bool block_ray = false,
                      const glm::mat4x3 &transform = glm::mat4x3(1.0f));

  glm::vec3 emission;
  int two_sided;
  int block_ray;
  glm::mat4x3 transform;

 private:
  Geometry *geometry_;
};

}  // namespace sparkium
