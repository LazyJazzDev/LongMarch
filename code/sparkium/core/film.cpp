#include "sparkium/core/film.h"

namespace sparkium {
Film::Film(Core *core, int width, int height)
    : core_(core), extent_{static_cast<uint32_t>(width), static_cast<uint32_t>(height)} {
}

void Film::Reset() {
}

int Film::GetWidth() const {
  return extent_.width;
}

int Film::GetHeight() const {
  return extent_.height;
}

void Film::Develop(graphics::Image *targ_image) {
}

}  // namespace sparkium
