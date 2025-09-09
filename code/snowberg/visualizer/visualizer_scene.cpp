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

void Scene::PyBind(pybind11::module_ &m) {
  pybind11::class_<Scene, std::shared_ptr<Scene>> scene(m, "Scene");
  scene.def("get_core", &Scene::GetCore);
  scene.def("add_entity", &Scene::AddEntity, pybind11::arg("entity"));
}

}  // namespace snowberg::visualizer
