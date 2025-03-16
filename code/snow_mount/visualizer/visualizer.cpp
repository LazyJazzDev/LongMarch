#include "snow_mount/visualizer/visualizer.h"

#include "visualizer_core.h"
#include "visualizer_film.h"
#include "visualizer_mesh.h"

namespace snow_mount::visualizer {

void PyBind(pybind11::module_ &m) {
  pybind11::enum_<FilmChannel> film_channel(m, "FilmChannel");
  film_channel.value("FILM_CHANNEL_EXPOSURE", FilmChannel::FILM_CHANNEL_EXPOSURE);
  film_channel.value("FILM_CHANNEL_ALBEDO", FilmChannel::FILM_CHANNEL_ALBEDO);
  film_channel.value("FILM_CHANNEL_POSITION", FilmChannel::FILM_CHANNEL_POSITION);
  film_channel.value("FILM_CHANNEL_NORMAL", FilmChannel::FILM_CHANNEL_NORMAL);
  film_channel.value("FILM_CHANNEL_DEPTH", FilmChannel::FILM_CHANNEL_DEPTH);
  film_channel.export_values();

  Core::PyBind(m);
  Mesh::PyBind(m);
  Film::PyBind(m);
}

}  // namespace snow_mount::visualizer
