#pragma once
#include "sparks/core/core_util.h"
#include "sparks/core/light.h"

namespace sparks {

class Entity {
 public:
  Entity(Core *core) : core_(core) {
  }
  virtual ~Entity() = default;
  virtual void Update(Scene *scene) = 0;

 protected:
  Core *core_;
};

class LightEntity : public Light {
 public:
  LightEntity(Core *core);
  graphics::Shader *SamplerShader() override;
  graphics::Buffer *SamplerData() override;

 private:
  std::unique_ptr<graphics::Shader> direct_lighting_sampler_;
  std::unique_ptr<graphics::Buffer> direct_lighting_sampler_data_;
};

class EntityGeometrySurface : public Entity {
 public:
  EntityGeometrySurface(Core *core,
                        Geometry *geometry,
                        Surface *surface,
                        const glm::mat4 &transformation = glm::mat4{1.0f});
  void Update(Scene *scene) override;

  void SetTransformation(const glm::mat4 &transformation) {
    transformation_ = transformation;
  }

 private:
  Geometry *geometry_{nullptr};
  Surface *surface_{nullptr};
  glm::mat4 transformation_;
};

class EntityGeometryLight : public Entity {
 public:
  EntityGeometryLight(Core *core, Geometry *geometry, const glm::vec3 &emission)
      : Entity(core), geometry_(geometry), emission(emission) {
  }

  glm::vec3 emission;
  void Update(Scene *scene) override;

 private:
  Geometry *geometry_;
};

}  // namespace sparks
