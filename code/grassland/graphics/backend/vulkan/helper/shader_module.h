#pragma once

#include "grassland/graphics/backend/vulkan/helper/native_types.h"

namespace grassland::graphics::backend::vulkan {
class ShaderModule {
 public:
  ShaderModule(const VulkanCore *device, VkShaderModule shader_module, const std::string &entry_point);

  ~ShaderModule();

  VkShaderModule Handle() const {
    return shader_module_;
  }

  const VulkanCore *Device() const {
    return device_;
  }

  const std::string &EntryPoint() const {
    return entry_point_;
  }

 private:
  const VulkanCore *device_{};
  VkShaderModule shader_module_{};
  std::string entry_point_;
};

struct HitGroup {
  ShaderModule *closest_hit_shader{nullptr};
  ShaderModule *any_hit_shader{nullptr};
  ShaderModule *intersection_shader{nullptr};
  bool procedure{false};
};

}  // namespace grassland::graphics::backend::vulkan
