#include "sparks/core/entity.h"

#include "scene.h"

namespace sparks {

EntityGeometryObject::EntityGeometryObject(Core *core,
                                           Geometry *geometry,
                                           Material *material,
                                           const glm::mat4 &transformation)
    : Entity(core) {
  geometry_ = geometry;
  material_ = material;
  transformation_ = transformation;
}

void EntityGeometryObject::Update(Scene *scene) {
  scene->RegisterInstance(scene->RegisterGeometry(geometry_), transformation_, scene->RegisterMaterial(material_));
}
}  // namespace sparks
