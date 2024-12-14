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
  virtual void AddInputBinding(uint32_t stride,
                               bool input_per_instance = false) = 0;
  virtual void AddInputAttribute(uint32_t binding,
                                 InputType type,
                                 uint32_t offset) = 0;
  virtual void BindShader(Shader *shader, ShaderType type) = 0;
  virtual void Finalize() = 0;
};

}  // namespace grassland::graphics
