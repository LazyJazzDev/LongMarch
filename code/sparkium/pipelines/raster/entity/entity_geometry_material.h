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
  std::unique_ptr<graphics::Program> render_program_;
  std::unique_ptr<graphics::Buffer> instance_buffer_;

  std::unique_ptr<graphics::Buffer> point_light_buffer_;
  std::unique_ptr<graphics::Shader> point_light_vs_;
  std::unique_ptr<graphics::Shader> point_light_ps_;
  std::unique_ptr<graphics::Program> point_light_program_;
};

}  // namespace sparkium::raster
