#pragma once
#include "sparks/core/core_util.h"

namespace sparks {

class Entity {
 public:
  Entity(Core *core) : core_(core) {
  }
  virtual ~Entity() {
  }
  virtual void Update(Scene *scene) = 0;

 protected:
  Core *core_;
};

class EntityGeometryObject : public Entity {
 public:
  EntityGeometryObject(Core *core,
                       Geometry *geometry,
                       Material *material,
                       const glm::mat4 &transformation = glm::mat4{1.0f});
  void Update(Scene *scene) override;

  void SetTransformation(const glm::mat4 &transformation) {
    transformation_ = transformation;
  }

 private:
  Geometry *geometry_{nullptr};
  Material *material_{nullptr};
  glm::mat4 transformation_;
};

}  // namespace sparks
