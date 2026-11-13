#pragma once

#include <algorithm>
#include <array>
#include <vector>

#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {

struct QueueSelection {
  uint32_t graphics_family{UINT32_MAX};
  uint32_t compute_family{UINT32_MAX};
  uint32_t graphics_count{};
  std::array<uint32_t, 3> indices{};
};

inline QueueSelection SelectQueues(const std::vector<VkQueueFamilyProperties> &families) {
  QueueSelection result;
  for (uint32_t i = 0; i < families.size(); ++i) {
    const auto required = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
    if (!families[i].queueCount || (families[i].queueFlags & required) != required)
      continue;
    if (result.graphics_family == UINT32_MAX || families[i].queueCount > families[result.graphics_family].queueCount)
      result.graphics_family = i;
    if (families[i].queueCount >= 3)
      break;
  }
  if (result.graphics_family == UINT32_MAX)
    return result;
  result.compute_family = result.graphics_family;
  for (uint32_t i = 0; i < families.size(); ++i) {
    if (families[i].queueCount && (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
        !(families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
      result.compute_family = i;
      break;
    }
  }
  const bool dedicated = result.compute_family != result.graphics_family;
  result.graphics_count = std::min(dedicated ? 2u : 3u, families[result.graphics_family].queueCount);
  result.indices = {0, std::min(1u, result.graphics_count - 1),
                    dedicated ? 0u : std::min(2u, result.graphics_count - 1)};
  return result;
}

}  // namespace grassland::rhi::backend::vulkan
