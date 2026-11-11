#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_interface_id.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::vulkan::IVulkanBuffer)
#include "grassland/rhi/backend/backend_buffer.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {
// Resource access only. Derived classes define allocation ownership (standalone or future pool leases).
class IVulkanBuffer : public IBackendBuffer {
 public:
  virtual VkBuffer NativeBuffer() const = 0;
};
}  // namespace grassland::rhi::backend::vulkan
