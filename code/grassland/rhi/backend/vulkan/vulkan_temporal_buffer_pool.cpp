#include "grassland/rhi/backend/vulkan/vulkan_temporal_buffer_pool.h"

#include "grassland/rhi/backend/vulkan/vulkan_device.h"

namespace grassland::rhi::backend::vulkan {

IVulkanTemporalBufferLease *IVulkanTemporalBufferLease::Create(VulkanTemporalBufferPool *pool,
                                                               IVulkanAllocatedBuffer *buffer,
                                                               size_t level) {
  return new VulkanTemporalBufferLeaseImpl(pool, buffer, level);
}

void IVulkanTemporalBufferLease::ReturnBuffer(VulkanTemporalBufferPool *pool,
                                              IVulkanAllocatedBuffer *buffer,
                                              size_t level) {
  pool->ReleaseBuffer(buffer, level);
}

VulkanTemporalBufferLeaseImpl::VulkanTemporalBufferLeaseImpl(VulkanTemporalBufferPool *pool,
                                                             IVulkanAllocatedBuffer *buffer,
                                                             size_t level)
    : pool_(pool),
      buffer_(buffer),
      level_(level) {
}

VulkanTemporalBufferLeaseImpl::~VulkanTemporalBufferLeaseImpl() {
  ReturnBuffer(pool_, buffer_, level_);
}

BufferDesc VulkanTemporalBufferLeaseImpl::GetDesc() const {
  return {VulkanTemporalBufferPool::CapacityForLevel(level_)};
}

VkBuffer VulkanTemporalBufferLeaseImpl::NativeBuffer() const {
  return buffer_->NativeBuffer();
}

bool VulkanTemporalBufferLeaseImpl::Write(const void *data, uint64_t size) {
  return buffer_->Write(data, size);
}

bool VulkanTemporalBufferLeaseImpl::Read(void *data, uint64_t size, uint64_t offset) {
  return buffer_->Read(data, size, offset);
}
}  // namespace grassland::rhi::backend::vulkan
