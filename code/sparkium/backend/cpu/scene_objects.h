#pragma once

#include "sparkium/backend/cpu/entity.h"
#include "sparkium/backend/cpu/geometry.h"
#include "sparkium/backend/cpu/material.h"
#include "sparkium/backend/cpu/texture.h"
#include "sparkium/core/camera.h"
#include "sparkium/core/core.h"
#include "sparkium/core/film.h"
#include "sparkium/core/scene.h"
#include "sparkium/entity/entities.h"
#include "sparkium/geometry/geometries.h"
#include "sparkium/material/materials.h"
#include "sparkium/scene/scene_definition.h"

namespace sparkium::backend::cpu {
class SceneObjects final {
 public:
  SceneObjects(Core *core, std::shared_ptr<const SceneDefinition> source);
  std::shared_ptr<const SceneDefinition> definition;
  std::map<const TextureData *, Texture> textures;
  std::map<std::string, Material> materials;
  std::map<std::string, Geometry> geometries;
  std::vector<Entity> entities;
  std::unique_ptr<Camera> camera;
  std::unique_ptr<Film> film;
  std::unique_ptr<Scene> scene;
};
}  // namespace sparkium::backend::cpu
