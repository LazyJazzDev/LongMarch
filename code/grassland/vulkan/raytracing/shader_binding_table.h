#pragma once
#include "grassland/vulkan/device.h"

namespace grassland::vulkan {

class ShaderBindingTable {
 public:
  explicit ShaderBindingTable(std::unique_ptr<Buffer> buffer,
                              VkDeviceAddress ray_gen_address,
                              VkDeviceAddress miss_address,
                              VkDeviceAddress hit_group_address,
                              VkDeviceAddress callable_address,
                              size_t miss_shader_count,
                              size_t hit_group_count,
                              size_t callable_shader_count);
  VkDeviceAddress GetRayGenDeviceAddress() const;
  VkDeviceAddress GetMissDeviceAddress() const;
  VkDeviceAddress GetHitGroupDeviceAddress() const;
  VkDeviceAddress GetCallableDeviceAddress() const;
  size_t MissShaderCount() const;
  size_t HitGroupCount() const;
  size_t CallableShaderCount() const;

 private:
  std::unique_ptr<Buffer> buffer_;
  VkDeviceAddress ray_gen_address_;
  VkDeviceAddress miss_address_;
  VkDeviceAddress hit_group_address_;
  VkDeviceAddress callable_address_;
  size_t miss_shader_count_{};
  size_t hit_group_count_{};
  size_t callable_shader_count_{};
};

}  // namespace grassland::vulkan
