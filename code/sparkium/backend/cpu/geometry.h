#pragma once
#include "sparkium/core/camera.h"
#include "sparkium/core/core.h"
#include "sparkium/core/film.h"
#include "sparkium/core/scene.h"
#include "sparkium/entity/entities.h"
#include "sparkium/geometry/geometries.h"
#include "sparkium/material/materials.h"
#include "sparkium/scene/scene_definition.h"

namespace sparkium::backend::cpu {
struct Geometry {
  Geometry(Core *core, std::shared_ptr<const GeometryDefinition> source);

  sparkium::Geometry *get() const {
    return object.get();
  }

  std::shared_ptr<const GeometryDefinition> source;
  std::unique_ptr<sparkium::Geometry> object;
};
}  // namespace sparkium::backend::cpu
