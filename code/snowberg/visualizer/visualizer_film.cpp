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

void Film::PyBind(pybind11::module_ &m) {
  pybind11::class_<Film, std::shared_ptr<Film>> film(m, "Film");
  film.def("__repr__", [](const Film &film) {
    return pybind11::str("Film({}x{})").format(film.Extent().width, film.Extent().height);
  });
  film.def("get_core", &Film::GetCore);
  film.def("extent", &Film::Extent);
  film.def("get_image", &Film::GetImage, pybind11::return_value_policy::reference,
           pybind11::arg("film_channel") = FILM_CHANNEL_EXPOSURE);
}

}  // namespace snowberg::visualizer
