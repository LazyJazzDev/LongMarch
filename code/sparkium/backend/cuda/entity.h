#pragma once
#include "sparkium/backend/common/scene_objects.h"

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
