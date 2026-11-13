#pragma once
#include <memory>

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
  rhi::ObjPtr<rhi::IBuffer> point_light_buffer_;
  rhi::ObjPtr<rhi::IShader> point_light_vs_;
  rhi::ObjPtr<rhi::IShader> point_light_ps_;
  std::shared_ptr<const rhi::ProgramDesc> point_light_description_;
  rhi::ObjPtr<rhi::IBindingGroup> point_light_group_;
};

}  // namespace sparkium::raster
