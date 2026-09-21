#pragma once
#include "sparkium/backend/cuda/entity.h"
#include "sparkium/backend/cuda/geometry.h"
#include "sparkium/backend/cuda/material.h"
#include "sparkium/backend/cuda/texture.h"

namespace sparkium::backend::cuda {
class SceneObjects final : public SceneObjectSet<Geometry, Texture, Material, Entity> {
 public:
  using SceneObjectSet::SceneObjectSet;
};
}  // namespace sparkium::backend::cuda
