#include "grassland/graphics/graphics.h"

namespace grassland::graphics {
void PybindModuleRegistration(py::module_ &m) {
  util::PybindModuleRegistration(m);
  py::classh<Core> c_core(m, "Core");
  py::classh<Core::Settings> c_core_settings(m, "CoreSettings");
  py::classh<Shader> c_shader(m, "Shader");
  py::classh<Program> c_program(m, "Program");
  py::classh<CommandContext> c_command_context(m, "CommandContext");
  py::classh<Buffer> c_buffer(m, "Buffer");
  py::classh<Image> c_image(m, "Image");
  py::classh<AccelerationStructure> c_acceleration_structure(m, "AccelerationStructure");
  py::classh<Sampler> c_sampler(m, "Sampler");
  py::classh<Window> c_window(m, "Window");

  Core::Settings::PybindClassRegistration(c_core_settings);
  Core::PybindClassRegistration(c_core);
  Shader::PybindClassRegistration(c_shader);
  Program::PybindClassRegistration(c_program);
  CommandContext::PybindClassRegistration(c_command_context);
  Buffer::PybindClassRegistration(c_buffer);
  Image::PybindClassRegistration(c_image);
  AccelerationStructure::PybindClassRegistration(c_acceleration_structure);
  Sampler::PybindClassRegistration(c_sampler);
  Window::PybindClassRegistration(c_window);
}
}  // namespace grassland::graphics
