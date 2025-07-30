#include "sparks/core/entity.h"

#include "scene.h"

namespace sparks {

EntityGeometrySurface::EntityGeometrySurface(Core *core,
                                             Geometry *geometry,
                                             Surface *surface,
                                             const glm::mat4 &transformation)
    : Entity(core) {
  geometry_ = geometry;
  surface_ = surface;
  transformation_ = transformation;
}

void EntityGeometrySurface::Update(Scene *scene) {
  scene->RegisterInstance(scene->RegisterGeometry(geometry_), transformation_, scene->RegisterSurface(surface_));
}
}  // namespace sparks
