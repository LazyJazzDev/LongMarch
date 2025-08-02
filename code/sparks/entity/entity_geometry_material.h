#pragma once
#include "sparks/core/entity.h"

namespace sparks {

class EntityGeometryMaterial : public Entity {
 public:
  EntityGeometryMaterial(Core *core,
                         Geometry *geometry,
                         Material *material,
                         const glm::mat4x3 &transformation = glm::mat4x3{1.0f});
  void Update(Scene *scene) override;

  void SetTransformation(const glm::mat4x3 &transformation) {
    transformation_ = transformation;
  }

 private:
  Geometry *geometry_{nullptr};
  Material *material_{nullptr};
  glm::mat4x3 transformation_;
  std::unique_ptr<graphics::Shader> closest_hit_shader_;
  std::unique_ptr<graphics::Shader> shadow_closest_hit_shader_;
};

}  // namespace sparks
