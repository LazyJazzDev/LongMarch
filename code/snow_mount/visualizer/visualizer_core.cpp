#include "snow_mount/visualizer/visualizer_core.h"

#include "snow_mount/visualizer/visualizer_film.h"
#include "snow_mount/visualizer/visualizer_mesh.h"

namespace snow_mount::visualizer {

Core::Core(graphics::Core *core) : core_(core) {
}

graphics::Core *Core::GraphicsCore() const {
  return core_;
}

std::shared_ptr<Core> Core::CreateCore(graphics::Core *graphics_core) {
  return std::shared_ptr<Core>(new Core(graphics_core));
}

std::shared_ptr<Mesh> Core::CreateMesh() {
  return std::shared_ptr<Mesh>{new Mesh(shared_from_this())};
}

std::shared_ptr<Film> Core::CreateFilm(int width, int height) {
  return std::shared_ptr<Film>{new Film(shared_from_this(), width, height)};
}

std::shared_ptr<Core> CreateCore(graphics::Core *graphics_core) {
  return Core::CreateCore(graphics_core);
}

void Core::PyBind(pybind11::module_ &m) {
  pybind11::class_<Core, std::shared_ptr<Core>> core_class(m, "Core");
  core_class.def(
      pybind11::init([](graphics::Core *graphics_core) { return std::shared_ptr<Core>(new Core(graphics_core)); }),
      pybind11::arg("graphics_core"), pybind11::keep_alive<1, 2>());
  core_class.def("create_mesh", &Core::CreateMesh, pybind11::keep_alive<0, 1>());
  core_class.def("create_film", &Core::CreateFilm, pybind11::keep_alive<0, 1>(), pybind11::arg("width"),
                 pybind11::arg("height"));
}

}  // namespace snow_mount::visualizer
