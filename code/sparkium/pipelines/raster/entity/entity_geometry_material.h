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

 private:
  sparkium::EntityGeometryMaterial &entity_;
  Geometry *geometry_{};
  Material *material_{};
  std::unique_ptr<graphics::Program> render_program_;
  std::unique_ptr<graphics::Buffer> instance_buffer_;
};

}  // namespace sparkium::raster
