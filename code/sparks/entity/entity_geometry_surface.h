#pragma once
#include "sparks/core/entity.h"

namespace sparks {

class EntityGeometrySurface : public Entity {
 public:
  EntityGeometrySurface(Core *core,
                        Geometry *geometry,
                        Surface *surface,
                        const glm::mat4x3 &transformation = glm::mat4x3{1.0f});
  void Update(Scene *scene) override;

  void SetTransformation(const glm::mat4x3 &transformation) {
    transformation_ = transformation;
  }

 private:
  Geometry *geometry_{nullptr};
  Surface *surface_{nullptr};
  glm::mat4x3 transformation_;
  std::unique_ptr<graphics::Shader> closest_hit_shader_;
};

}  // namespace sparks
