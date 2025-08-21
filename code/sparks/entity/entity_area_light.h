// #pragma once
// #include "sparks/core/entity.h"
// #include "sparks/entity/entity_geometry_light.h"
// #include "sparks/geometry/geometry_mesh.h"
//
// namespace sparks {
// class EntityAreaLight : public Entity {
//  public:
//   EntityAreaLight(Core *core,
//                   const glm::vec3 &emission = {1.0f, 1.0f, 1.0f},
//                   float size = 1.0f,
//                   const glm::vec3 &position = {0.0f, 0.0f, 0.0f},
//                   const glm::vec3 &direction = {0.0f, 0.0f, 1.0f},
//                   const glm::vec3 &up = {0.0f, 1.0f, 0.0f});
//   void Update(Scene *scene) override;
//
//   glm::vec3 emission{1.0f, 1.0f, 1.0f};
//   float size{1.0f};
//   glm::vec3 position{0.0f, 0.0f, 0.0f};
//   glm::vec3 direction{0.0f, -1.0f, 0.0f};
//   glm::vec3 up{0.0f, 1.0f, 0.0f};
//
//  private:
//   std::unique_ptr<GeometryMesh> mesh_;
//   std::unique_ptr<EntityGeometryLight> geometry_light_;
// };
// }  // namespace sparks
