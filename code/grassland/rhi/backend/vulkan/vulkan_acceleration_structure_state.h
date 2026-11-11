#pragma once
#include "grassland/rhi/backend/vulkan/vulkan_acceleration_structure.h"

namespace grassland::rhi::backend::vulkan {
struct VulkanAccelerationStructureState final {
  VulkanAccelerationStructureState();
  ~VulkanAccelerationStructureState();
  IVulkanDevice *device_{};
  VkBuffer buffer_{};
  VmaAllocation allocation_{};
  VkAccelerationStructureKHR structure_{};
  VkQueryPool query_{};
  VkBuildAccelerationStructureFlagsKHR flags_{};
  uint64_t address_{};
  uint64_t build_scratch_{}, update_scratch_{};
  bool allow_update_{}, allow_compaction_{}, compact_{};
};
}  // namespace grassland::rhi::backend::vulkan
