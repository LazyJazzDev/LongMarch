#include "xue_shan/solver/solver.h"

namespace XS::solver {
void PyBind(pybind11::module_ &m) {
  element::PyBind(m);
  ObjectPack::PyBind(m);
  Scene::PyBind(m);
  RigidObject::PyBind(m);
}
}  // namespace XS::solver
