#pragma once
#include "sparkium/pipelines/raytracing/core/entity.h"
#include "sparkium/pipelines/raytracing/light/light_geometry_material.h"

namespace sparkium::raytracing {

class EntityGeometryMaterial : public Entity {
 public:
  EntityGeometryMaterial(sparkium::EntityGeometryMaterial &entity);
  void Update(Scene *scene) override;

 private:
  void PrepareHitGroups();

  sparkium::EntityGeometryMaterial &entity_;
  Geometry *geometry_{nullptr};
  Material *material_{nullptr};
  std::unique_ptr<LightGeometryMaterial> light_geom_mat_;
  rhi::ObjPtr<rhi::IShader> closest_hit_shader_;
  rhi::ObjPtr<rhi::IShader> shadow_closest_hit_shader_;
  InstanceHitGroups hit_groups_;
};

}  // namespace sparkium::raytracing
