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

Core *Scene::GetCore() const {
  return core_;
}

void Scene::AddEntity(Entity *entity) {
  if (!entities_.count(entity))
    entities_.insert({entity, {true, next_entity_order_++}});
}

void Scene::DeleteEntity(Entity *entity) {
  entities_.erase(entity);
}

void Scene::SetEntityActive(Entity *entity, bool active) {
  entities_.at(entity).active = active;
}

const std::map<Entity *, Scene::EntityStatus> &Scene::GetEntities() const {
  return entities_;
}

}  // namespace sparkium
