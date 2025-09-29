#include "sparkium/entity/entity_geometry_material.h"

#include "sparkium/core/core.h"
#include "sparkium/core/geometry.h"
#include "sparkium/core/material.h"
#include "sparkium/core/scene.h"
#include "sparkium/geometry/geometries.h"
#include "sparkium/material/materials.h"

namespace sparkium {

EntityGeometryMaterial::EntityGeometryMaterial(Core *core,
                                               Geometry *geometry,
                                               Material *material,
                                               const glm::mat4x3 &transformation)
    : Entity(core), geometry_(geometry), material_(material), transform(transformation) {
}

void EntityGeometryMaterial::SetTransformation(const glm::mat4x3 &transformation) {
  transform = transformation;
}

glm::mat4x3 EntityGeometryMaterial::GetTransformation() const {
  return transform;
}

Geometry *EntityGeometryMaterial::GetGeometry() const {
  return geometry_;
}

Material *EntityGeometryMaterial::GetMaterial() const {
  return material_;
}

}  // namespace sparkium
