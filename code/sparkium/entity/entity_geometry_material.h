#pragma once
#include "sparkium/core/entity.h"

namespace sparkium {

class EntityGeometryMaterial : public Entity {
 public:
  EntityGeometryMaterial(Core *core,
                         Geometry *geometry,
                         Material *material,
                         const glm::mat4x3 &transformation = glm::mat4x3{1.0f});

  void SetTransformation(const glm::mat4x3 &transformation);

 private:
  glm::mat4x3 transformation_;
  Geometry *geometry_{nullptr};
  Material *material_{nullptr};
};

}  // namespace sparkium
