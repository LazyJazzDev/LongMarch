#pragma once

#include "sparkium/core/camera.h"
#include "sparkium/core/core.h"
#include "sparkium/core/film.h"
#include "sparkium/core/scene.h"
#include "sparkium/entity/entities.h"
#include "sparkium/geometry/geometries.h"
#include "sparkium/material/materials.h"
#include "sparkium/scene/scene_definition.h"

namespace sparkium::detail {
// Execution objects for the existing pipelines. Never exposed by the scene API.
// Each renderer owns one instance; the immutable definition outlives its resources.
class SceneInstance {
 public:
  SceneInstance(Core *core, std::shared_ptr<const SceneDefinition> definition);
  std::shared_ptr<const SceneDefinition> definition;
  std::map<const TextureData *, std::unique_ptr<graphics::Image>> images;
  std::map<std::string, std::unique_ptr<Material>> materials;
  std::map<std::string, std::unique_ptr<Geometry>> geometries;
  std::vector<std::unique_ptr<Entity>> entities;
  std::unique_ptr<Camera> camera;
  std::unique_ptr<Film> film;
  std::unique_ptr<Scene> scene;
};
}  // namespace sparkium::detail
