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
struct Texture {
  Texture(Core *core, std::shared_ptr<const TextureData> source);

  graphics::Image *get() const {
    return image.get();
  }

  std::shared_ptr<const TextureData> source;
  std::unique_ptr<graphics::Image> image;
};
}  // namespace sparkium::backend::cpu
