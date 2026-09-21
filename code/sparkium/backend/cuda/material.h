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
struct Material {
  Material(std::unique_ptr<sparkium::Material> translated, const MaterialDefinition *definition)
      : source(definition),
        object(std::move(translated)) {
  }

  sparkium::Material *get() const {
    return object.get();
  }

  const MaterialDefinition *source;  // Owned by the immutable scene snapshot.
  std::unique_ptr<sparkium::Material> object;
};
}  // namespace sparkium::backend::cuda
