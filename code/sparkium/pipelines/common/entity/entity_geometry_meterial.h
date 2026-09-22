#pragma once
#include "sparkium/pipelines/common/core/entity.h"
#include "sparkium/pipelines/common/light/light_geometry_material.h"

namespace sparkium::render_shared {

class EntityGeometryMaterial : public Entity {
 public:
  EntityGeometryMaterial(sparkium::EntityGeometryMaterial &entity);
  void Update(Scene *scene) override;

 private:
  sparkium::EntityGeometryMaterial &entity_;
  Geometry *geometry_{nullptr};
  Material *material_{nullptr};
  std::unique_ptr<LightGeometryMaterial> light_geom_mat_;
  std::unique_ptr<graphics::Shader> closest_hit_shader_;
  std::unique_ptr<graphics::Shader> shadow_closest_hit_shader_;
  InstanceHitGroups hit_groups_;
};

}  // namespace sparkium::render_shared
