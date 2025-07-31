#include "sparks/entity/entity_geometry_surface.h"

#include "sparks/core/scene.h"

namespace sparks {

EntityGeometrySurface::EntityGeometrySurface(Core *core,
                                             Geometry *geometry,
                                             Surface *surface,
                                             const glm::mat4x3 &transformation)
    : Entity(core) {
  geometry_ = geometry;
  surface_ = surface;
  transformation_ = transformation;
}

void EntityGeometrySurface::Update(Scene *scene) {
  scene->RegisterInstance(scene->RegisterGeometry(geometry_), transformation_, scene->RegisterSurface(surface_));
}

}  // namespace sparks
