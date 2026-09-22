#pragma once
#include "sparkium/pipelines/realtime/core/entity.h"
#include "sparkium/pipelines/realtime/light/light_point.h"

namespace sparkium::realtime {

class EntityPointLight : public Entity {
 public:
  EntityPointLight(sparkium::EntityPointLight &entity);

  void Update(Scene *scene) override;

 private:
  sparkium::EntityPointLight &entity_;
  LightPoint light_point_;
};

}  // namespace sparkium::realtime
