#pragma once
#include "sparkium/pipelines/raster/core/entity.h"

namespace sparkium::raster {

class EntityGeometryMaterial : public Entity {
 public:
  struct InstanceData {
    glm::mat4 model;
    glm::mat4 inv_model;
    glm::mat4 normal_matrix;
  } instance_data;

  EntityGeometryMaterial(sparkium::EntityGeometryMaterial &entity);

  void Update(Scene *scene) override;

  struct PointLightData {
    glm::vec3 emission;
    glm::vec3 position;
  } point_light_data;

 private:
  sparkium::EntityGeometryMaterial &entity_;
  Geometry *geometry_{};
  Material *material_{};
  rhi::ObjPtr<rhi::IProgram> render_program_;
  rhi::ObjPtr<rhi::IBindingGroup> render_group_;
  rhi::ObjPtr<rhi::IBuffer> instance_buffer_;

  rhi::ObjPtr<rhi::IBuffer> point_light_buffer_;
  rhi::ObjPtr<rhi::IShader> point_light_vs_;
  rhi::ObjPtr<rhi::IShader> point_light_ps_;
  rhi::ObjPtr<rhi::IProgram> point_light_program_;
  rhi::ObjPtr<rhi::IBindingGroup> point_light_group_;
};

}  // namespace sparkium::raster
