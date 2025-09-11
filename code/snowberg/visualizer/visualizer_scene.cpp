#include "snowberg/visualizer/visualizer_scene.h"

#include "snowberg/visualizer/visualizer_core.h"
#include "snowberg/visualizer/visualizer_entity.h"

namespace snowberg::visualizer {
Scene::Scene(const std::shared_ptr<Core> &core) : core_(core) {
}

std::shared_ptr<Core> Scene::GetCore() const {
  return core_;
}

uint64_t Scene::AddEntity(const std::shared_ptr<Entity> &entity) {
  uint64_t id = entity_next_id_++;
  entities_[id] = entity;
  return id;
}

}  // namespace snowberg::visualizer
