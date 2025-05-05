#include "snow_mount/draw/draw_texture.h"

#include "snow_mount/draw/draw_core.h"

namespace snow_mount::draw {

Texture::Texture(Core *core, int width, int height) : core_(core) {
  core_->GraphicsCore()->CreateImage(width, height, graphics::IMAGE_FORMAT_R8G8B8A8_UNORM, &image_);
}

}  // namespace snow_mount::draw
