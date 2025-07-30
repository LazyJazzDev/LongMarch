#pragma once
#include "sparks/core/entity.h"

namespace sparks {

class EntityGeometryLight : public Entity {
 public:
  EntityGeometryLight(Core *core, Geometry *geometry, const glm::vec3 &emission)
      : Entity(core), geometry_(geometry), emission(emission) {
  }

  glm::vec3 emission;
  void Update(Scene *scene) override;

 private:
  Geometry *geometry_;
};

}  // namespace sparks
