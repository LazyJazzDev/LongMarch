#pragma once
#include "snow_mount/visualizer/visualizer_util.h"

namespace snow_mount::visualizer {
class Film {
  Film(const std::shared_ptr<Core> &core, int width, int height);
  friend class Core;

 public:
  graphics::Extent2D Extent() const;

  graphics::Image *GetImage(FilmChannel film_channel);

  static void PyBind(pybind11::module_ &m);

 private:
  std::shared_ptr<Core> core_;
  std::unique_ptr<graphics::Image> images_[FILM_CHANNEL_COUNT];
};
}  // namespace snow_mount::visualizer
