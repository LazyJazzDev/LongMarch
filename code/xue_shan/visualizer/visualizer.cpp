#include "xue_shan/visualizer/visualizer.h"

#include "xue_shan/visualizer/visualizer_camera.h"
#include "xue_shan/visualizer/visualizer_core.h"
#include "xue_shan/visualizer/visualizer_entity.h"
#include "xue_shan/visualizer/visualizer_film.h"
#include "xue_shan/visualizer/visualizer_mesh.h"
#include "xue_shan/visualizer/visualizer_scene.h"

namespace XS::visualizer {

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

}  // namespace XS::visualizer
