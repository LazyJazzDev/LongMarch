#include "cao_di/vulkan/raytracing/shader_binding_table.h"

#include "cao_di/vulkan/buffer.h"

namespace CD::vulkan {

ShaderBindingTable::ShaderBindingTable(std::unique_ptr<Buffer> buffer,
                                       VkDeviceAddress ray_gen_address,
                                       VkDeviceAddress miss_address,
                                       VkDeviceAddress hit_group_address,
                                       VkDeviceAddress callable_address,
                                       size_t miss_shader_count,
                                       size_t hit_group_count,
                                       size_t callable_shader_count)
    : buffer_(std::move(buffer)),
      ray_gen_address_(ray_gen_address),
      miss_address_(miss_address),
      hit_group_address_(hit_group_address),
      callable_address_(callable_address),
      miss_shader_count_(miss_shader_count),
      hit_group_count_(hit_group_count),
      callable_shader_count_(callable_shader_count) {
}

VkDeviceAddress ShaderBindingTable::GetRayGenDeviceAddress() const {
  return ray_gen_address_;
}

VkDeviceAddress ShaderBindingTable::GetMissDeviceAddress() const {
  return miss_address_;
}

VkDeviceAddress ShaderBindingTable::GetHitGroupDeviceAddress() const {
  return hit_group_address_;
}

VkDeviceAddress ShaderBindingTable::GetCallableDeviceAddress() const {
  return callable_address_;
}

size_t ShaderBindingTable::MissShaderCount() const {
  return miss_shader_count_;
}

size_t ShaderBindingTable::HitGroupCount() const {
  return hit_group_count_;
}

size_t ShaderBindingTable::CallableShaderCount() const {
  return callable_shader_count_;
}

}  // namespace CD::vulkan
