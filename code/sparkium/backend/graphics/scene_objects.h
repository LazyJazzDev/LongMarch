#pragma once
#include "sparkium/backend/graphics/entity.h"
#include "sparkium/backend/graphics/geometry.h"
#include "sparkium/backend/graphics/material.h"
#include "sparkium/backend/graphics/texture.h"

namespace sparkium::backend::graphics_backend {
class SceneObjects final : public SceneObjectSet<Geometry, Texture, Material, Entity> {
 public:
  using SceneObjectSet::SceneObjectSet;
};
}  // namespace sparkium::backend::graphics_backend
