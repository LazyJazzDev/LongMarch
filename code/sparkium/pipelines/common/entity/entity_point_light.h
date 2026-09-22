#pragma once
#include "sparkium/pipelines/common/core/entity.h"
#include "sparkium/pipelines/common/light/light_point.h"

namespace sparkium::render_shared {

class EntityPointLight : public Entity {
 public:
  EntityPointLight(sparkium::EntityPointLight &entity);

  void Update(Scene *scene) override;

 private:
  sparkium::EntityPointLight &entity_;
  LightPoint light_point_;
};

}  // namespace sparkium::render_shared
