#pragma once
#include "sparkium/pipelines/raytracing/core/entity.h"
#include "sparkium/pipelines/raytracing/light/light_point.h"

namespace sparkium::raytracing {

class EntityPointLight : public Entity {
 public:
  EntityPointLight(sparkium::EntityPointLight &entity);

  void Update(Scene *scene) override;

 private:
  sparkium::EntityPointLight &entity_;
  LightPoint light_point_;
};

}  // namespace sparkium::raytracing
