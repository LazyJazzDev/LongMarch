#pragma once
#include "sparkium/pipelines/raster/core/entity.h"

namespace sparkium::raster {

class EntityPointLight : public Entity {
 public:
  EntityPointLight(sparkium::EntityPointLight &entity);

  void Update(Scene *scene) override;

  struct PointLightData {
    glm::vec3 emission;
    glm::vec3 position;
  } point_light_data;

 private:
  sparkium::EntityPointLight &entity_;
  std::unique_ptr<graphics::Buffer> point_light_buffer_;
  std::unique_ptr<graphics::Shader> point_light_vs_;
  std::unique_ptr<graphics::Shader> point_light_ps_;
  std::unique_ptr<graphics::Program> point_light_program_;
};

}  // namespace sparkium::raster
