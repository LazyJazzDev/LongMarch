#pragma once
#include "sparks/core/entity.h"
#include "sparks/light/light_point.h"

namespace XH {

class EntityPointLight : public Entity {
 public:
  EntityPointLight(Core *core,
                   const glm::vec3 &position = {},
                   const glm::vec3 &color = {1.0f, 1.0f, 1.0f},
                   float strength = 0.0f);

  void Update(Scene *scene) override;

  glm::vec3 &position = light_point_.position;
  glm::vec3 &color = light_point_.color;
  float &strength = light_point_.strength;

 private:
  LightPoint light_point_;
};

}  // namespace XH
