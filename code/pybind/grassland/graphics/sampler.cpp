#include "pybind/grassland/graphics/graphics.h"

namespace grassland::graphics::pybind {
void RegisterSampler(py::classh<Sampler> &c) {
  c.def("__repr__", [](Sampler *sampler) { return py::str("Sampler()"); });
}
}  // namespace grassland::graphics::pybind
