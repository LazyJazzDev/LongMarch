#pragma once
#include "snow_mount/visualizer/visualizer_util.h"

namespace snow_mount::visualizer {

class Core : public std::enable_shared_from_this<Core> {
  Core(graphics::Core *core);

 public:
  graphics::Core *GraphicsCore() const;

  static std::shared_ptr<Core> CreateCore(graphics::Core *graphics_core);

  std::shared_ptr<Mesh> CreateMesh();

  std::shared_ptr<Film> CreateFilm(int width, int height);

  static void PyBind(pybind11::module_ &m);

 private:
  graphics::Core *core_;
};

std::shared_ptr<Core> CreateCore(graphics::Core *graphics_core);

}  // namespace snow_mount::visualizer
