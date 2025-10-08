#include "grassland/graphics/sampler.h"

namespace grassland::graphics {

#if defined(LONGMARCH_PYTHON_ENABLED)
void Sampler::PybindClassRegistration(py::classh<Sampler> &c) {
  c.def("__repr__", [](Sampler *sampler) { return py::str("Sampler()"); });
}
#endif

}  // namespace grassland::graphics
