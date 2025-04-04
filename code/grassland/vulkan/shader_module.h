#pragma once

#include "grassland/vulkan/device.h"

namespace grassland::vulkan {
class ShaderModule {
 public:
  ShaderModule(const class Device *device, VkShaderModule shader_module, const std::string &entry_point);

  ~ShaderModule();

  VkShaderModule Handle() const {
    return shader_module_;
  }

  const class Device *Device() const {
    return device_;
  }

  const std::string &EntryPoint() const {
    return entry_point_;
  }

 private:
  const class Device *device_{};
  VkShaderModule shader_module_{};
  std::string entry_point_;
};

CompiledShaderBlob CompileGLSLToSPIRV(const std::string &glsl_code, VkShaderStageFlagBits shader_stage);
}  // namespace grassland::vulkan
