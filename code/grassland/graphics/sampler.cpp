#include "grassland/graphics/sampler.h"

namespace grassland::graphics {

void Sampler::PybindClassRegistration(py::classh<Sampler> &c) {
  c.def("__repr__", [](Sampler *sampler) { return py::str("Sampler()"); });
}

}  // namespace grassland::graphics
