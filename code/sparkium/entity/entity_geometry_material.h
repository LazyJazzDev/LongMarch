#pragma once
#include "sparkium/core/entity.h"

namespace sparkium {

class EntityGeometryMaterial : public Entity {
 public:
  EntityGeometryMaterial(Core *core,
                         Geometry *geometry,
                         Material *material,
                         const glm::mat4x3 &transformation = glm::mat4x3{1.0f});

  glm::mat4x3 transform;
  bool raster_light{true};

  void SetTransformation(const glm::mat4x3 &transformation);

  glm::mat4x3 GetTransformation() const;
  Geometry *GetGeometry() const;
  Material *GetMaterial() const;

 private:
  Geometry *geometry_{nullptr};
  Material *material_{nullptr};
};

}  // namespace sparkium
