#pragma once
#include "grassland/rhi/backend/vulkan/vulkan_buffer.h"

namespace grassland::rhi::backend::vulkan {
class VulkanUploadPool;
class VulkanAllocatedBuffer;

// Borrows pool-owned storage. The pool and its Device must outlive every lease.
class VulkanUploadBufferLease final : public VulkanBuffer {
 public:
  VulkanUploadBufferLease(VulkanUploadPool *pool, VulkanAllocatedBuffer *buffer);
  ~VulkanUploadBufferLease() override;
  BufferDesc GetDesc() const override;
  VkBuffer NativeBuffer() const override;
  bool Write(const void *data, uint64_t size);

 private:
  VulkanUploadPool *pool_;
  VulkanAllocatedBuffer *buffer_;
};
}  // namespace grassland::rhi::backend::vulkan
