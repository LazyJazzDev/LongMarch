#pragma once
#include "sparkium/pipelines/realtime/core/entity.h"
#include "sparkium/pipelines/realtime/light/light_geometry_material.h"

namespace sparkium::realtime {

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

}  // namespace sparkium::realtime
