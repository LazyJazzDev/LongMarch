#include "grassland/graphics/program.h"

#include <stdexcept>

#include "grassland/graphics/core.h"

namespace grassland::graphics {

namespace {
class SourceShader : public Shader {
 public:
  explicit SourceShader(const ShaderCode &code) : code(code) {
  }

  std::string EntryPoint() const override {
    return code.EntryPoint();
  }

  ShaderCode code;
};
}  // namespace

Shader *ProgramShaderBindings::StoreShaderCode(const ShaderCode &code) {
  shader_codes_.push_back(std::make_unique<SourceShader>(code));
  return shader_codes_.back().get();
}

bool ProgramShaderBindings::IsShaderCode(Shader *shader) const {
  return dynamic_cast<SourceShader *>(shader) != nullptr;
}

Shader *ProgramShaderBindings::ResolveShader(Core *core, Shader *shader) {
  auto source = dynamic_cast<SourceShader *>(shader);
  if (!source)
    return shader;
  auto &compiled = compiled_shaders_[shader];
  if (!compiled)
    compiled = source->code.Compile(core, resource_bindings_);
  return compiled.get();
}

void ProgramShaderBindings::RecordResourceBinding(ResourceType type, int count) {
  if (!compiled_shaders_.empty())
    throw std::logic_error("resource layout is frozen after shader compilation");
  if (type < RESOURCE_TYPE_UNIFORM_BUFFER || type > RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER || count <= 0 ||
      (type == RESOURCE_TYPE_ACCELERATION_STRUCTURE && count != 1))
    throw std::invalid_argument("invalid resource binding type or count");
  resource_bindings_.emplace_back(type, count);
}

void RayTracingProgram::AddHitGroup(const ShaderCode &closest_hit,
                                    const ShaderCode *any_hit,
                                    const ShaderCode *intersection,
                                    bool procedure) {
  AddHitGroup(StoreShaderCode(closest_hit), any_hit ? StoreShaderCode(*any_hit) : nullptr,
              intersection ? StoreShaderCode(*intersection) : nullptr, procedure);
}

#if defined(LONGMARCH_PYTHON_ENABLED)
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
  c.def("bind_shader", py::overload_cast<Shader *, ShaderType>(&Program::BindShader), py::arg("shader"),
        py::arg("type"), "Bind a shader to the program", py::keep_alive<1, 2>{});
  c.def("finalize", &Program::Finalize, "Finalize the program");
  c.def("__repr__", [](Program *program) { return py::str("Program()"); });
}
#endif

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

#if defined(LONGMARCH_PYTHON_ENABLED)
void ComputeProgram::PybindClassRegistration(py::classh<ComputeProgram> &c) {
  c.def("add_resource_binding", &ComputeProgram::AddResourceBinding, py::arg("type"), py::arg("count"),
        "Add a resource binding");
  c.def("finalize", &ComputeProgram::Finalize, "Finalize the compute program");
  c.def("__repr__", [](ComputeProgram *program) { return py::str("ComputeProgram()"); });
}
#endif

#if defined(LONGMARCH_PYTHON_ENABLED)
void RayTracingProgram::PybindClassRegistration(py::classh<RayTracingProgram> &c) {
  c.def("add_resource_binding", &RayTracingProgram::AddResourceBinding, py::arg("type"), py::arg("count"),
        "Add a resource binding");
  c.def("add_ray_gen_shader", py::overload_cast<Shader *>(&RayTracingProgram::AddRayGenShader),
        py::arg("ray_gen_shader"), "Add a ray generation shader", py::keep_alive<1, 2>{});
  c.def("add_miss_shader", py::overload_cast<Shader *>(&RayTracingProgram::AddMissShader), py::arg("miss_shader"),
        "Add a miss shader", py::keep_alive<1, 2>{});
  c.def("add_hit_group", py::overload_cast<Shader *, Shader *, Shader *, bool>(&RayTracingProgram::AddHitGroup),
        py::arg("closest_hit_shader"), py::arg("any_hit_shader") = nullptr, py::arg("intersection_shader") = nullptr,
        py::arg("procedure") = false, "Add a hit group", py::keep_alive<1, 2>{}, py::keep_alive<1, 3>{},
        py::keep_alive<1, 4>{});
  c.def("add_callable_shader", py::overload_cast<Shader *>(&RayTracingProgram::AddCallableShader),
        py::arg("callable_shader"), "Add a callable shader", py::keep_alive<1, 2>{});
  c.def("finalize",
        py::overload_cast<const std::vector<int32_t> &, const std::vector<int32_t> &, const std::vector<int32_t> &>(
            &RayTracingProgram::Finalize),
        py::arg("miss_shader_indices"), py::arg("hit_group_indices"), py::arg("callable_shader_indices"),
        "Finalize the ray tracing program with specific indices");
  c.def("finalize", py::overload_cast<>(&RayTracingProgram::Finalize), "Finalize the ray tracing program");
  c.def("__repr__", [](RayTracingProgram *program) { return py::str("RayTracingProgram()"); });
}
#endif

}  // namespace grassland::graphics
