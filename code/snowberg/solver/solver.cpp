#include "snowberg/solver/solver.h"

namespace snowberg::solver {
void PyBind(pybind11::module_ &m) {
  element::PyBind(m);
  ObjectPack::PyBind(m);
  Scene::PyBind(m);
  RigidObject::PyBind(m);
}
}  // namespace snowberg::solver
