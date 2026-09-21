#pragma once
#include "sparkium/backend/common/scene_objects.h"

namespace sparkium::backend::cuda {
struct Geometry {
  Geometry(Core *core, std::shared_ptr<const GeometryDefinition> source);

  sparkium::Geometry *get() const {
    return object.get();
  }

  std::shared_ptr<const GeometryDefinition> source;
  std::unique_ptr<sparkium::Geometry> object;
};
}  // namespace sparkium::backend::cuda
