#include "pybind/pybind.h"

namespace grassland::graphics::pybind {
void RegisterGraphics(py::module_ &m) {
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
#if defined(LONGMARCH_CUDA_RUNTIME)
  // Buffer is a virtual base: pointer adjustment must go through pybind11 casts.
  py::classh<CUDABuffer, Buffer> c_cuda_buffer(m, "CUDABuffer", py::multiple_inheritance());
#endif
  py::classh<Image> c_image(m, "Image");
  py::classh<Sampler> c_sampler(m, "Sampler");
  py::classh<AccelerationStructure> c_acceleration_structure(m, "AccelerationStructure");

  // Context and Window classes
  py::classh<CommandContext> c_command_context(m, "CommandContext");
  py::classh<Window> c_window(m, "Window");

  // Register all classes
  RegisterTypes(m);

  py::enum_<MagnifyPhase> magnify_phase(m, "MagnifyPhase");
  magnify_phase.value("MAGNIFY_PHASE_BEGIN", MagnifyPhase::kBegin, "Magnify Phase: Begin");
  magnify_phase.value("MAGNIFY_PHASE_UPDATE", MagnifyPhase::kUpdate, "Magnify Phase: Update");
  magnify_phase.value("MAGNIFY_PHASE_END", MagnifyPhase::kEnd, "Magnify Phase: End");
  magnify_phase.value("MAGNIFY_PHASE_CANCEL", MagnifyPhase::kCancel, "Magnify Phase: Cancel");
  magnify_phase.export_values();

  RegisterCoreSettings(c_core_settings);
  RegisterCore(c_core);
  RegisterShader(c_shader);
  RegisterProgram(c_program);
  RegisterComputeProgram(c_compute_program);
  RegisterRayTracingProgram(c_raytracing_program);
  RegisterCommandContext(c_command_context);
  RegisterBuffer(c_buffer);
  RegisterImage(c_image);
  RegisterAccelerationStructure(c_acceleration_structure);
  RegisterSampler(c_sampler);
  RegisterWindow(c_window);

#if defined(LONGMARCH_CUDA_RUNTIME)
  c_cuda_buffer.doc() = "Device buffer shared with CUDA; cuda_ptr() is a device pointer in the core's CUDA device";
  c_cuda_buffer.def("cuda_ptr", [](CUDABuffer &buffer) {
    void *ptr = nullptr;
    buffer.GetCUDAMemoryPointer(&ptr);
    return reinterpret_cast<uintptr_t>(ptr);
  });
#endif

  py::classh<FrameProfile> c_frame_profile(m, "FrameProfile");
  c_frame_profile.doc() = "Opt-in frame profiling; GPU timestamps currently require Vulkan. Read gpu_ms after wait_gpu";
  c_frame_profile.def(py::init<Core *, bool>(), py::arg("core"), py::arg("gpu_timestamps") = true,
                      py::keep_alive<1, 2>());
  c_frame_profile.def("begin", &FrameProfile::Begin, py::arg("gpu_timestamps") = true);
  c_frame_profile.def("begin_gpu", &FrameProfile::BeginGpu, py::arg("command_context"), py::arg("name"));
  c_frame_profile.def("end_gpu", &FrameProfile::EndGpu, py::arg("command_context"), py::arg("index"));
  c_frame_profile.def("finish", &FrameProfile::Finish);
  c_frame_profile.def_readonly("gpu_ms", &FrameProfile::gpu_ms);
  c_frame_profile.def_readonly("cpu_ms", &FrameProfile::cpu_ms);

  auto m_imgui = m.def_submodule("imgui", "Dear ImGui widgets for graphics windows");
  RegisterImGui(m_imgui);
}
}  // namespace grassland::graphics::pybind
