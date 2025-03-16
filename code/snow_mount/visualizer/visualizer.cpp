#include "snow_mount/visualizer/visualizer.h"

#include "snow_mount/visualizer/visualizer_core.h"
#include "snow_mount/visualizer/visualizer_film.h"
#include "snow_mount/visualizer/visualizer_mesh.h"
#include "visualizer_camera.h"
#include "visualizer_entity.h"
#include "visualizer_scene.h"

namespace snow_mount::visualizer {

void PyBind(pybind11::module_ &m) {
  pybind11::enum_<FilmChannel> film_channel(m, "FilmChannel");
  film_channel.value("FILM_CHANNEL_EXPOSURE", FilmChannel::FILM_CHANNEL_EXPOSURE);
  film_channel.value("FILM_CHANNEL_ALBEDO", FilmChannel::FILM_CHANNEL_ALBEDO);
  film_channel.value("FILM_CHANNEL_POSITION", FilmChannel::FILM_CHANNEL_POSITION);
  film_channel.value("FILM_CHANNEL_NORMAL", FilmChannel::FILM_CHANNEL_NORMAL);
  film_channel.value("FILM_CHANNEL_DEPTH", FilmChannel::FILM_CHANNEL_DEPTH);
  film_channel.export_values();

  Material::PyBind(m);
  Core::PyBind(m);
  Mesh::PyBind(m);
  Film::PyBind(m);
  Scene::PyBind(m);
  Camera::PyBind(m);
  Entity::PyBind(m);
}

}  // namespace snow_mount::visualizer
