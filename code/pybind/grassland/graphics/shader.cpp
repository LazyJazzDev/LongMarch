#include "pybind/grassland/graphics/graphics.h"

namespace grassland::graphics::pybind {
void RegisterShader(py::classh<Shader> &c) {
  c.def("entry_point", &Shader::EntryPoint, "Get the shader entry point");
  c.def("__repr__", [](Shader *shader) { return py::str("Shader(entry_point='{}')").format(shader->EntryPoint()); });
}
}  // namespace grassland::graphics::pybind
