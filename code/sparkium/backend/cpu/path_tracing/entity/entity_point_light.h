#pragma once
#include "sparkium/backend/cpu/path_tracing/core/entity.h"
#include "sparkium/backend/cpu/path_tracing/light/light_point.h"

namespace sparkium::cpu_tracing {

class EntityPointLight : public Entity {
 public:
  EntityPointLight(sparkium::EntityPointLight &entity);

  void Update(Scene *scene) override;

 private:
  sparkium::EntityPointLight &entity_;
  LightPoint light_point_;
};

}  // namespace sparkium::cpu_tracing
