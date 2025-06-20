#pragma once
#include "grassland/graphics/graphics_util.h"

namespace grassland::graphics {

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
  virtual void Finalize() = 0;
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

}  // namespace grassland::graphics
