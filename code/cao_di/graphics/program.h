#pragma once
#include "cao_di/graphics/graphics_util.h"

namespace CD::graphics {

class Shader {
 public:
  virtual ~Shader() = default;
};

class Program {
 public:
  virtual ~Program() = default;
  virtual void AddInputBinding(uint32_t stride, bool input_per_instance = false) = 0;
  virtual void AddInputAttribute(uint32_t binding, InputType type, uint32_t offset) = 0;
  virtual void AddResourceBinding(ResourceType type, int count) = 0;
  virtual void SetCullMode(CullMode mode) = 0;
  virtual void SetBlendState(int target_id, const BlendState &state) = 0;
  virtual void BindShader(Shader *shader, ShaderType type) = 0;
  virtual void Finalize() = 0;
};

class ComputeProgram {
 public:
  virtual ~ComputeProgram() = default;
  virtual void AddResourceBinding(ResourceType type, int count) = 0;
  virtual void Finalize() = 0;
};

class RayTracingProgram {
 public:
  virtual ~RayTracingProgram() = default;
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
};

struct HitGroup {
  Shader *closest_hit_shader{nullptr};
  Shader *any_hit_shader{nullptr};
  Shader *intersection_shader{nullptr};
  bool procedure{false};
};

CompiledShaderBlob CompileShader(const std::string &source_code,
                                 const std::string &entry_point,
                                 const std::string &target,
                                 const std::vector<std::string> &args = {});

CompiledShaderBlob CompileShader(const VirtualFileSystem &vfs,
                                 const std::string &source_file,
                                 const std::string &entry_point,
                                 const std::string &target,
                                 const std::vector<std::string> &args = {});

}  // namespace CD::graphics
