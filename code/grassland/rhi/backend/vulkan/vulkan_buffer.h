#pragma once
#include "grassland/rhi/backend/backend_buffer.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {
class VulkanDevice;

// Resource access only. Derived classes define allocation ownership (standalone or future pool leases).
class VulkanBuffer : public Buffer {
 public:
  LONGMARCH_RHI_QUERY(VulkanBuffer, Buffer)

  virtual VkBuffer NativeBuffer() const = 0;

  Device *Owner() const override;

 protected:
  VulkanDevice *device_;
  VulkanBuffer(VulkanDevice *device, const BufferDesc &desc);
};
}  // namespace grassland::rhi::backend::vulkan
