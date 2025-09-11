#pragma once
#include "snowberg/visualizer/visualizer_util.h"

namespace snowberg::visualizer {
class Film {
  Film(const std::shared_ptr<Core> &core, int width, int height);
  friend class Core;

 public:
  std::shared_ptr<Core> GetCore() const;

  graphics::Extent2D Extent() const;

  graphics::Image *GetImage(FilmChannel film_channel);

 private:
  std::shared_ptr<Core> core_;
  std::unique_ptr<graphics::Image> images_[FILM_CHANNEL_COUNT];
};
}  // namespace snowberg::visualizer
