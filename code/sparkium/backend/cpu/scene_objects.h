#pragma once
#include "sparkium/backend/cpu/entity.h"
#include "sparkium/backend/cpu/geometry.h"
#include "sparkium/backend/cpu/material.h"
#include "sparkium/backend/cpu/texture.h"

namespace sparkium::backend::cpu {
class SceneObjects final : public SceneObjectSet<Geometry, Texture, Material, Entity> {
 public:
  using SceneObjectSet::SceneObjectSet;
};
}  // namespace sparkium::backend::cpu
