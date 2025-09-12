#include "grassland/graphics/graphics.h"

namespace grassland::graphics {
void PybindModuleRegistration(py::module_ &m) {
  util::PybindModuleRegistration(m);
  Core::PybindModuleRegistration(m);
  CommandContext::PybindModuleRegistration(m);
  Buffer::PybindModuleRegistration(m);
  Image::PybindModuleRegistration(m);
  AccelerationStructure::PybindModuleRegistration(m);
  Shader::PybindModuleRegistration(m);
  Program::PybindModuleRegistration(m);
  Sampler::PybindModuleRegistration(m);
  Window::PybindModuleRegistration(m);
}
}  // namespace grassland::graphics
