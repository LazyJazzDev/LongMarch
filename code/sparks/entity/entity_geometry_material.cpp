#include "sparks/entity/entity_geometry_material.h"

#include "sparks/core/core.h"
#include "sparks/core/geometry.h"
#include "sparks/core/material.h"
#include "sparks/core/scene.h"

namespace sparks {

EntityGeometryMaterial::EntityGeometryMaterial(Core *core,
                                               Geometry *geometry,
                                               Material *material,
                                               const glm::mat4x3 &transformation)
    : Entity(core), light_geom_mat_(core, geometry, material, transformation) {
  geometry_ = geometry;
  material_ = material;
  transformation_ = transformation;
}

void EntityGeometryMaterial::Update(Scene *scene) {
  auto geom_reg = scene->RegisterGeometry(geometry_);
  auto mat_reg = scene->RegisterMaterial(material_);
  int32_t light_index = scene->RegisterLight(&light_geom_mat_);
  int instance_index = scene->RegisterInstance(geom_reg, transformation_, mat_reg, light_index);
  scene->LightCustomIndex(light_index) = instance_index;
}

}  // namespace sparks
