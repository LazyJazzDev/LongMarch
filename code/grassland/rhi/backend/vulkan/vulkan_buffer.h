#pragma once
#include "grassland/rhi/backend/vulkan/vulkan_util.h"
#include "grassland/rhi/rhi_buffer.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::vulkan::IVulkanBuffer,
                        grassland::rhi::IObject,
                        "edd00360-77f8-46d3-9659-0c06d6616086")

namespace grassland::rhi::backend::vulkan {
// Resource access only. Derived classes define allocation ownership (standalone or future pool leases).
class IVulkanBuffer : public IObject {
 public:
  virtual IVulkanDevice *Owner() const = 0;
  virtual VkBuffer NativeBuffer() const = 0;
};
}  // namespace grassland::rhi::backend::vulkan
