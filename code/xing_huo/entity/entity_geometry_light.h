#pragma once
#include "xing_huo/core/entity.h"
#include "xing_huo/entity/entity_geometry_material.h"
#include "xing_huo/material/material_light.h"

namespace XH {

class EntityGeometryLight : public Entity {
 public:
  EntityGeometryLight(Core *core,
                      Geometry *geometry,
                      const glm::vec3 &emission,
                      bool two_sided = false,
                      bool block_ray = false,
                      const glm::mat4x3 &transform = glm::mat4x3(1.0f));

  void Update(Scene *scene) override;

  glm::vec3 emission;
  int two_sided;
  int block_ray;
  glm::mat4x3 transform;

 private:
  Geometry *geometry_;
  MaterialLight material_light_;
  std::unique_ptr<EntityGeometryMaterial> entity_;
};

}  // namespace XH
