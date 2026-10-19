#include "grassland/rhi/backend/vulkan/vulkan_upload_buffer_lease.h"

#include "grassland/rhi/backend/vulkan/vulkan_device.h"
#include "grassland/rhi/backend/vulkan/vulkan_upload_pool.h"

namespace grassland::rhi::backend::vulkan {
VulkanUploadBufferLease::VulkanUploadBufferLease(VulkanUploadPool *pool, VulkanAllocatedBuffer *buffer)
    : VulkanBuffer(static_cast<VulkanDevice *>(buffer->Owner()), buffer->GetDesc()),
      pool_(pool),
      buffer_(buffer) {
}

VulkanUploadBufferLease::~VulkanUploadBufferLease() {
  pool_->ReleaseBuffer(buffer_);
}

BufferDesc VulkanUploadBufferLease::GetDesc() const {
  return buffer_->GetDesc();
}

VkBuffer VulkanUploadBufferLease::NativeBuffer() const {
  return buffer_->NativeBuffer();
}

bool VulkanUploadBufferLease::Write(const void *data, uint64_t size) {
  return buffer_->Write(data, size);
}
}  // namespace grassland::rhi::backend::vulkan
