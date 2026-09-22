#pragma once
#include <map>

#include "grassland/graphics/graphics_util.h"
#include "grassland/graphics/shader_code.h"

namespace grassland::graphics {

// Shared source ownership and per-program compiled shader lifetime.
class ProgramShaderBindings {
 public:
  virtual ~ProgramShaderBindings() = default;

 protected:
  Shader *StoreShaderCode(const ShaderCode &code);
  Shader *ResolveShader(Core *core, Shader *shader);
  bool IsShaderCode(Shader *shader) const;
  void RecordResourceBinding(ResourceType type, int count);
  std::vector<std::pair<ResourceType, int>> resource_bindings_;

 private:
  std::vector<std::unique_ptr<Shader>> shader_codes_;
  std::map<Shader *, std::unique_ptr<Shader>> compiled_shaders_;
};

class Program : public ProgramShaderBindings {
 public:
  virtual ~Program() = default;
  virtual void AddInputBinding(uint32_t stride, bool input_per_instance = false) = 0;
  virtual void AddInputAttribute(uint32_t binding, InputType type, uint32_t offset) = 0;
  virtual void AddResourceBinding(ResourceType type, int count) = 0;
  virtual void SetCullMode(CullMode mode) = 0;
  virtual void SetBlendState(int target_id, const BlendState &state) = 0;
  virtual void BindShader(Shader *shader, ShaderType type) = 0;

  void BindShader(const ShaderCode &code, ShaderType type) {
    BindShader(StoreShaderCode(code), type);
  }

  virtual void Finalize() = 0;

#if defined(LONGMARCH_PYTHON_ENABLED)
  static void PybindClassRegistration(py::classh<Program> &c);
#endif
};

class ComputeProgram : public ProgramShaderBindings {
 public:
  virtual ~ComputeProgram() = default;
  virtual void BindShader(Shader *shader) = 0;

  void BindShader(const ShaderCode &code) {
    BindShader(StoreShaderCode(code));
  }

  virtual void AddResourceBinding(ResourceType type, int count) = 0;
  virtual void Finalize() = 0;

#if defined(LONGMARCH_PYTHON_ENABLED)
  static void PybindClassRegistration(py::classh<ComputeProgram> &c);
#endif
};

class RayTracingProgram : public ProgramShaderBindings {
 public:
  virtual ~RayTracingProgram() = default;

  void AddRayGenShader(const ShaderCode &code) {
    AddRayGenShader(StoreShaderCode(code));
  }

  void AddMissShader(const ShaderCode &code) {
    AddMissShader(StoreShaderCode(code));
  }

  void AddCallableShader(const ShaderCode &code) {
    AddCallableShader(StoreShaderCode(code));
  }

  void AddHitGroup(const ShaderCode &closest_hit,
                   const ShaderCode *any_hit = nullptr,
                   const ShaderCode *intersection = nullptr,
                   bool procedure = false);
  virtual void AddResourceBinding(ResourceType type, int count) = 0;
  virtual void AddRayGenShader(Shader *ray_gen_shader) = 0;
  virtual void AddMissShader(Shader *miss_shader) = 0;
  void AddHitGroup(Shader *closest_hit_shader,
                   Shader *any_hit_shader = nullptr,
                   Shader *intersection_shader = nullptr,
                   bool procedure = false);
  virtual void AddHitGroup(HitGroup hit_group) = 0;
  virtual void AddCallableShader(Shader *callable_shader) = 0;
  virtual void Finalize(const std::vector<int32_t> &miss_shader_indices,
                        const std::vector<int32_t> &hit_group_indices,
                        const std::vector<int32_t> &callable_shader_indices) = 0;
  virtual void Finalize() = 0;

#if defined(LONGMARCH_PYTHON_ENABLED)
  static void PybindClassRegistration(py::classh<RayTracingProgram> &c);
#endif
};

struct HitGroup {
  Shader *closest_hit_shader{nullptr};
  Shader *any_hit_shader{nullptr};
  Shader *intersection_shader{nullptr};
  bool procedure{false};
};

}  // namespace grassland::graphics
