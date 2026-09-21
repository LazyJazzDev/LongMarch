#pragma once
#include "sparkium/backend/common/scene_objects.h"

namespace sparkium::backend::graphics_backend {
struct Texture {
  Texture(Core *core, std::shared_ptr<const TextureData> source);

  graphics::Image *get() const {
    return image.get();
  }

  std::shared_ptr<const TextureData> source;
  std::unique_ptr<graphics::Image> image;
};
}  // namespace sparkium::backend::graphics_backend
