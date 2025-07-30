#pragma once
#include "sparks/core/core_util.h"
#include "sparks/core/light.h"

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

class LightEntity : public Light {
 public:
  LightEntity(Core *core);

 private:
  std::unique_ptr<graphics::Shader> direct_lighting_sampler_;
  std::unique_ptr<graphics::Buffer> direct_lighting_sampling_data_;
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

class EntityGeometryLight : public Entity {
 public:
  EntityGeometryLight(Core *core, Geometry *geometry, const glm::vec3 &emission)
      : Entity(core), geometry_(geometry), emission(emission) {
  }

  glm::vec3 emission;

 private:
  Geometry *geometry_;
};

}  // namespace sparks
