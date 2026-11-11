#pragma once
#include "grassland/rhi/backend/backend_buffer.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::vulkan::IVulkanBuffer)

namespace grassland::rhi::backend::vulkan {
// Resource access only. Derived classes define allocation ownership (standalone or future pool leases).
class IVulkanBuffer : public IBackendBuffer {
 public:
  virtual VkBuffer NativeBuffer() const = 0;
};
}  // namespace grassland::rhi::backend::vulkan
