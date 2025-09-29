#include "sparkium/core/scene.h"

#include <numeric>

#include "sparkium/core/camera.h"
#include "sparkium/core/core.h"
#include "sparkium/core/entity.h"
#include "sparkium/core/film.h"
#include "sparkium/core/geometry.h"
#include "sparkium/core/material.h"

namespace sparkium {
Scene::Scene(Core *core) : core_(core) {
}

void Scene::Render(Camera *camera, Film *film) {
}

void Scene::AddEntity(Entity *entity) {
  entities_.insert({entity, {}});
}

void Scene::DeleteEntity(Entity *entity) {
  entities_.erase(entity);
}

void Scene::SetEntityActive(Entity *entity, bool active) {
  entities_.at(entity).active = active;
}

}  // namespace sparkium
