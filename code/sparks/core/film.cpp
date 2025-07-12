#include "sparks/core/film.h"

namespace sparks {
Film::Film(Core *core, int width, int height) : core_(core) {
  core_->GraphicsCore()->CreateImage(width, height, graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT, &accumulated_color_);
  core_->GraphicsCore()->CreateImage(width, height, graphics::IMAGE_FORMAT_R32_SINT, &accumulated_samples_);
}

}  // namespace sparks
