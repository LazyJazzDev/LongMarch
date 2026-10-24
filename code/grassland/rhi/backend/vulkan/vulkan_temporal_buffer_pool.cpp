#include "grassland/rhi/backend/vulkan/vulkan_temporal_buffer_pool.h"

#include "grassland/rhi/backend/vulkan/vulkan_device.h"

namespace grassland::rhi::backend::vulkan {
VulkanTemporalBufferLease::VulkanTemporalBufferLease(VulkanTemporalBufferPool *pool,
                                                     VulkanAllocatedBuffer *buffer,
                                                     size_t level)
    : VulkanBuffer(static_cast<VulkanDevice *>(buffer->Owner()), buffer->GetDesc()),
      pool_(pool),
      buffer_(buffer),
      level_(level) {
}

VulkanTemporalBufferLease::~VulkanTemporalBufferLease() {
  pool_->ReleaseBuffer(buffer_, level_);
}

BufferDesc VulkanTemporalBufferLease::GetDesc() const {
  return buffer_->GetDesc();
}

VkBuffer VulkanTemporalBufferLease::NativeBuffer() const {
  return buffer_->NativeBuffer();
}

bool VulkanTemporalBufferLease::Write(const void *data, uint64_t size) {
  return buffer_->Write(data, size);
}

bool VulkanTemporalBufferLease::Read(void *data, uint64_t size, uint64_t offset) {
  return buffer_->Read(data, size, offset);
}
}  // namespace grassland::rhi::backend::vulkan
