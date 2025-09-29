#pragma once
#include "sparkium/core/entity.h"
#include "sparkium/entity/entity_geometry_light.h"
#include "sparkium/geometry/geometry_mesh.h"

namespace sparkium {
class EntityAreaLight : public Entity {
 public:
  EntityAreaLight(Core *core,
                  const glm::vec3 &emission = {1.0f, 1.0f, 1.0f},
                  float size = 1.0f,
                  const glm::vec3 &position = {0.0f, 0.0f, 0.0f},
                  const glm::vec3 &direction = {0.0f, 0.0f, 1.0f},
                  const glm::vec3 &up = {0.0f, 1.0f, 0.0f});

  glm::vec3 emission{1.0f, 1.0f, 1.0f};
  float size{1.0f};
  glm::vec3 position{0.0f, 0.0f, 0.0f};
  glm::vec3 direction{0.0f, -1.0f, 0.0f};
  glm::vec3 up{0.0f, 1.0f, 0.0f};
};
}  // namespace sparkium
