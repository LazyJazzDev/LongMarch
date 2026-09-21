#pragma once
#include "sparkium/core/camera.h"
#include "sparkium/core/core.h"
#include "sparkium/core/film.h"
#include "sparkium/core/scene.h"
#include "sparkium/entity/entities.h"
#include "sparkium/geometry/geometries.h"
#include "sparkium/material/materials.h"
#include "sparkium/scene/scene_definition.h"

namespace sparkium::backend::cuda {
struct Entity {
  Entity(std::unique_ptr<sparkium::Entity> translated,
         const std::variant<InstanceDefinition, PointLightDefinition> *definition)
      : source(definition),
        object(std::move(translated)) {
  }

  sparkium::Entity *get() const {
    return object.get();
  }

  const std::variant<InstanceDefinition, PointLightDefinition> *source;  // Owned by the immutable scene snapshot.
  std::unique_ptr<sparkium::Entity> object;
};
}  // namespace sparkium::backend::cuda
