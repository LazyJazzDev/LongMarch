#include "grassland/graphics/program.h"

namespace grassland::graphics {

void Program::PybindClassRegistration(py::classh<Program> &c) {
  c.def("add_input_binding", &Program::AddInputBinding, py::arg("stride"), py::arg("input_per_instance") = false,
        "Add an input binding");
  c.def("add_input_attribute", &Program::AddInputAttribute, py::arg("binding"), py::arg("type"), py::arg("offset"),
        "Add an input attribute");
  c.def("add_resource_binding", &Program::AddResourceBinding, py::arg("type"), py::arg("count"),
        "Add a resource binding");
  c.def("set_cull_mode", &Program::SetCullMode, py::arg("mode"), "Set cull mode");
  c.def("set_blend_state", &Program::SetBlendState, py::arg("target_id"), py::arg("state"),
        "Set blend state for a render target");
  c.def("bind_shader", &Program::BindShader, py::arg("shader"), py::arg("type"), "Bind a shader to the program",
        py::keep_alive<1, 2>{});
  c.def("finalize", &Program::Finalize, "Finalize the program");
  c.def("__repr__", [](Program *program) { return py::str("Program()"); });
}

void RayTracingProgram::AddHitGroup(Shader *closest_hit_shader,
                                    Shader *any_hit_shader,
                                    Shader *intersection_shader,
                                    bool procedure) {
  HitGroup hit_group{
      closest_hit_shader,
      any_hit_shader,
      intersection_shader,
      procedure,
  };
  AddHitGroup(hit_group);
}

void ComputeProgram::PybindClassRegistration(py::classh<ComputeProgram> &c) {
  c.def("add_resource_binding", &ComputeProgram::AddResourceBinding, py::arg("type"), py::arg("count"),
        "Add a resource binding");
  c.def("finalize", &ComputeProgram::Finalize, "Finalize the compute program");
  c.def("__repr__", [](ComputeProgram *program) { return py::str("ComputeProgram()"); });
}

void RayTracingProgram::PybindClassRegistration(py::classh<RayTracingProgram> &c) {
  c.def("add_resource_binding", &RayTracingProgram::AddResourceBinding, py::arg("type"), py::arg("count"),
        "Add a resource binding");
  c.def("add_ray_gen_shader", &RayTracingProgram::AddRayGenShader, py::arg("ray_gen_shader"),
        "Add a ray generation shader", py::keep_alive<1, 2>{});
  c.def("add_miss_shader", &RayTracingProgram::AddMissShader, py::arg("miss_shader"), "Add a miss shader",
        py::keep_alive<1, 2>{});
  c.def("add_hit_group", py::overload_cast<Shader *, Shader *, Shader *, bool>(&RayTracingProgram::AddHitGroup),
        py::arg("closest_hit_shader"), py::arg("any_hit_shader") = nullptr, py::arg("intersection_shader") = nullptr,
        py::arg("procedure") = false, "Add a hit group", py::keep_alive<1, 2>{}, py::keep_alive<1, 3>{},
        py::keep_alive<1, 4>{});
  c.def("add_callable_shader", &RayTracingProgram::AddCallableShader, py::arg("callable_shader"),
        "Add a callable shader", py::keep_alive<1, 2>{});
  c.def("finalize",
        py::overload_cast<const std::vector<int32_t> &, const std::vector<int32_t> &, const std::vector<int32_t> &>(
            &RayTracingProgram::Finalize),
        py::arg("miss_shader_indices"), py::arg("hit_group_indices"), py::arg("callable_shader_indices"),
        "Finalize the ray tracing program with specific indices");
  c.def("finalize", py::overload_cast<>(&RayTracingProgram::Finalize), "Finalize the ray tracing program");
  c.def("__repr__", [](RayTracingProgram *program) { return py::str("RayTracingProgram()"); });
}

}  // namespace grassland::graphics
