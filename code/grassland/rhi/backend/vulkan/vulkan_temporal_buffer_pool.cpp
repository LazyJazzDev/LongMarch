#include "grassland/rhi/backend/vulkan/vulkan_temporal_buffer_pool.h"

#include "grassland/rhi/backend/vulkan/vulkan_device.h"

namespace grassland::rhi::backend::vulkan {
class VulkanTemporalBufferLeaseImpl final : public Object<rhi::IBuffer, IVulkanTemporalBufferLease> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(IBuffer, IVulkanTemporalBufferLease, IBuffer, IBackendBuffer, IVulkanBuffer)

  VulkanTemporalBufferLeaseImpl(VulkanTemporalBufferPool *pool, IVulkanAllocatedBuffer *buffer, size_t level);
  ~VulkanTemporalBufferLeaseImpl() override;
  BufferDesc GetDesc() const override;
  VkBuffer NativeBuffer() const override;
  bool Read(void *data, uint64_t size, uint64_t offset = 0);
  bool Write(const void *data, uint64_t size);

 private:
  VulkanTemporalBufferPool *pool_;
  IVulkanAllocatedBuffer *buffer_;
  size_t level_;

 public:
  IBackendDevice *Owner() const override {
    return buffer_->Owner();
  }
};

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
  return rhi::QueryInterfacePtr<IBuffer>(buffer_)->GetDesc();
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

LONGMARCH_RHI_UUID(grassland::rhi::backend::vulkan::IVulkanTemporalBufferLease, "b8b91e0e-bc72-4ea2-b918-d6135648577f")
