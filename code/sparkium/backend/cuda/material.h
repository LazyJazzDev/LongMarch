#pragma once
#include "sparkium/backend/common/scene_objects.h"

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
