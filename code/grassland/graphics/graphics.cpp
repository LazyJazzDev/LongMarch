#include "grassland/graphics/graphics.h"

namespace grassland::graphics {
void PybindModuleRegistration(py::module_ &m) {
  util::PybindModuleRegistration(m);

  // Core classes
  py::classh<Core> c_core(m, "Core");
  py::classh<Core::Settings> c_core_settings(m, "CoreSettings");

  // Shader classes
  py::classh<Shader> c_shader(m, "Shader");

  // Program classes
  py::classh<Program> c_program(m, "Program");
  py::classh<ComputeProgram> c_compute_program(m, "ComputeProgram");
  py::classh<RayTracingProgram> c_raytracing_program(m, "RayTracingProgram");

  // Resource classes
  py::classh<Buffer> c_buffer(m, "DeviceBuffer");
  py::classh<Image> c_image(m, "Image");
  py::classh<Sampler> c_sampler(m, "Sampler");
  py::classh<AccelerationStructure> c_acceleration_structure(m, "AccelerationStructure");

  // Context and Window classes
  py::classh<CommandContext> c_command_context(m, "CommandContext");
  py::classh<Window> c_window(m, "Window");

  // Register all classes
  Core::Settings::PybindClassRegistration(c_core_settings);
  Core::PybindClassRegistration(c_core);
  Shader::PybindClassRegistration(c_shader);
  Program::PybindClassRegistration(c_program);
  ComputeProgram::PybindClassRegistration(c_compute_program);
  RayTracingProgram::PybindClassRegistration(c_raytracing_program);
  CommandContext::PybindClassRegistration(c_command_context);
  Buffer::PybindClassRegistration(c_buffer);
  Image::PybindClassRegistration(c_image);
  AccelerationStructure::PybindClassRegistration(c_acceleration_structure);
  Sampler::PybindClassRegistration(c_sampler);
  Window::PybindClassRegistration(c_window);
}
}  // namespace grassland::graphics
