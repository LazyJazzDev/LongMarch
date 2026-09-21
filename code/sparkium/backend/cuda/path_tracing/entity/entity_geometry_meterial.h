#pragma once
#include "sparkium/backend/cuda/path_tracing/core/entity.h"
#include "sparkium/backend/cuda/path_tracing/light/light_geometry_material.h"

namespace sparkium::cuda_tracing {

class EntityGeometryMaterial : public Entity {
 public:
  EntityGeometryMaterial(sparkium::EntityGeometryMaterial &entity);
  void Update(Scene *scene) override;

 private:
  sparkium::EntityGeometryMaterial &entity_;
  Geometry *geometry_{nullptr};
  Material *material_{nullptr};
  std::unique_ptr<LightGeometryMaterial> light_geom_mat_;
};

}  // namespace sparkium::cuda_tracing
