#pragma once

#include "sparkium/core/camera.h"
#include "sparkium/core/core.h"
#include "sparkium/core/film.h"
#include "sparkium/core/scene.h"
#include "sparkium/entity/entities.h"
#include "sparkium/geometry/geometries.h"
#include "sparkium/material/materials.h"
#include "sparkium/scene/scene_definition.h"

namespace sparkium::backend {
// Common camera/film state. Concrete backends own their semantic resource sets.
class SceneObjects {
 public:
  explicit SceneObjects(std::shared_ptr<const SceneDefinition> source) : definition(std::move(source)) {
  }

  virtual ~SceneObjects() = default;
  std::shared_ptr<const SceneDefinition> definition;
  std::unique_ptr<Camera> camera;
  std::unique_ptr<Film> film;
  std::unique_ptr<Scene> scene;
};

// Translation is shared; storage, texture upload and geometry preparation are
// selected by concrete semantic types supplied by each backend.
template <class GeometryObject, class TextureObject, class MaterialObject, class EntityObject>
class SceneObjectSet : public SceneObjects {
 public:
  SceneObjectSet(Core *core, std::shared_ptr<const SceneDefinition> source);

  ~SceneObjectSet() override {
    scene.reset();
  }

  std::map<const TextureData *, TextureObject> textures;
  std::map<std::string, MaterialObject> materials;
  std::map<std::string, GeometryObject> geometries;
  std::vector<EntityObject> entities;
};
}  // namespace sparkium::backend

#include "sparkium/backend/common/scene_objects_impl.h"
