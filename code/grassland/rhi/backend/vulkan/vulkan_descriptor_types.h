#pragma once
#include "grassland/rhi/backend/vulkan/vulkan_util.h"
#include "grassland/rhi/rhi_binding_group.h"

namespace grassland::rhi::backend::vulkan {
namespace detail {
inline constexpr VkDescriptorType descriptor_types[] = {
    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
    VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_SAMPLER};

inline size_t Kind(BindingType type) {
  const size_t kinds[] = {0, 1, 1, 1, 1, 2, 3, 4};
  return kinds[type];
}

inline VkDescriptorType Type(BindingType type) {
  return descriptor_types[Kind(type)];
}
}  // namespace detail
}  // namespace grassland::rhi::backend::vulkan
