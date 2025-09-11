#include "snowberg/visualizer/visualizer_film.h"

#include "snowberg/visualizer/visualizer_core.h"

namespace snowberg::visualizer {

Film::Film(const std::shared_ptr<Core> &core, int width, int height) : core_(core) {
  core_->GraphicsCore()->CreateImage(width, height, FilmChannelImageFormat(FILM_CHANNEL_EXPOSURE),
                                     &images_[FILM_CHANNEL_EXPOSURE]);
}

std::shared_ptr<Core> Film::GetCore() const {
  return core_;
}

graphics::Extent2D Film::Extent() const {
  return images_[FILM_CHANNEL_EXPOSURE]->Extent();
}

graphics::Image *Film::GetImage(FilmChannel film_channel) {
  if (!images_[film_channel]) {
    core_->GraphicsCore()->CreateImage(Extent().width, Extent().height, FilmChannelImageFormat(film_channel),
                                       &images_[film_channel]);
  }
  return images_[film_channel].get();
}

}  // namespace snowberg::visualizer
