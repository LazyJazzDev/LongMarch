#include "snowberg/draw/draw_texture.h"

#include "snowberg/draw/draw_core.h"

namespace snowberg::draw {

Texture::Texture(Core *core, int width, int height) : core_(core) {
  core_->GraphicsCore()->CreateImage(width, height, graphics::IMAGE_FORMAT_R8G8B8A8_UNORM, &image_);
}

}  // namespace snowberg::draw
