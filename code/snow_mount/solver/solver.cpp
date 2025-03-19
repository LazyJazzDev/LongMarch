#include "snow_mount/solver/solver.h"

namespace snow_mount::solver {
void PyBind(pybind11::module_ &m) {
  element::PyBind(m);
  ObjectPack::PyBind(m);
  Scene::PyBind(m);
}
}  // namespace snow_mount::solver
