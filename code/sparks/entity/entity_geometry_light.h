#pragma once
#include "sparks/core/entity.h"
#include "sparks/light/light_geometry_surface.h"
#include "sparks/surface/surface_light.h"

namespace sparks {

class EntityGeometryLight : public Entity {
 public:
  EntityGeometryLight(Core *core,
                      Geometry *geometry,
                      const glm::vec3 &emission,
                      bool two_sided = false,
                      bool block_ray = false,
                      const glm::mat4x3 &transform = glm::mat4x3(1.0f));

  void Update(Scene *scene) override;

  glm::vec3 emission;
  int two_sided;
  int block_ray;

 private:
  Geometry *geometry_;
  SurfaceLight surface_light_;
  LightGeometrySurface light_geom_surf_;
  glm::mat4x3 transformation_;
};

}  // namespace sparks
