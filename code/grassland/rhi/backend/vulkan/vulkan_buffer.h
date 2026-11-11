#pragma once
#include "grassland/rhi/backend/backend_buffer.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::vulkan::IVulkanBuffer,
                        grassland::rhi::backend::IBackendBuffer,
                        "edd00360-77f8-46d3-9659-0c06d6616086")

namespace grassland::rhi::backend::vulkan {
// Resource access only. Derived classes define allocation ownership (standalone or future pool leases).
class IVulkanBuffer : public IBackendBuffer {
 public:
  virtual VkBuffer NativeBuffer() const = 0;
};
}  // namespace grassland::rhi::backend::vulkan
