#pragma once
#include "grassland/rhi/backend/backend_download_token.h"
#include "grassland/rhi/backend/vulkan/vulkan_temporal_buffer_pool.h"

namespace grassland::rhi::backend::vulkan {
class VulkanDownloadToken final : public DownloadToken {
 public:
  VulkanDownloadToken(ObjectPtr<VulkanTemporalBufferLease> lease, uint64_t size);
  VulkanDownloadToken(ObjectPtr<VulkanTemporalBufferLease> lease, const DownloadLayout &layout);
  ~VulkanDownloadToken() override;

 protected:
  bool CopyData(void *destination) const override;

 private:
  ObjectPtr<VulkanTemporalBufferLease> lease_;
  DownloadLayout layout_;
};
}  // namespace grassland::rhi::backend::vulkan
