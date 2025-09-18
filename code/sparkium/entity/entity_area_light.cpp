#include "sparkium/entity/entity_area_light.h"

#include <glm/ext/matrix_transform.hpp>

namespace sparkium {
EntityAreaLight::EntityAreaLight(Core *core,
                                 const glm::vec3 &emission,
                                 float size,
                                 const glm::vec3 &position,
                                 const glm::vec3 &direction,
                                 const glm::vec3 &up)
    : Entity(core), emission(emission), position(position), direction(direction), size(size), up(up) {
  uint32_t indices[] = {0, 2, 1, 0, 3, 2};
  Vector3<float> vertices[] = {{-1.0f, -1.0f, 0.0f}, {1.0f, -1.0f, 0.0f}, {1.0f, 1.0f, 0.0f}, {-1.0f, 1.0f, 0.0f}};
  Mesh<> mesh(4, 6, indices, vertices);
  mesh_ = std::make_unique<GeometryMesh>(core, mesh);
  geometry_light_ = std::make_unique<EntityGeometryLight>(
      core, mesh_.get(), emission, false, false,
      glm::inverse(glm::lookAt(position, position + direction, up)) * glm::scale(glm::mat4{1.0f}, glm::vec3{size}));
}

void EntityAreaLight::Update(Scene *scene) {
  const glm::mat4x3 transform =
      glm::inverse(glm::lookAt(position, position + direction, up)) * glm::scale(glm::mat4{1.0f}, glm::vec3{size});
  geometry_light_->emission = emission;
  geometry_light_->transform = transform;
  geometry_light_->Update(scene);
}
}  // namespace sparkium
