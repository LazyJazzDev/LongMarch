#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend::vulkan {
class VulkanDownloadToken;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::vulkan::VulkanDownloadToken)
#include "grassland/rhi/backend/backend_download_token.h"
#include "grassland/rhi/backend/vulkan/vulkan_temporal_buffer_pool.h"

namespace grassland::rhi::backend::vulkan {
class VulkanDownloadToken final : public DownloadToken {
 public:
  LONGMARCH_RHI_QUERY(VulkanDownloadToken, DownloadToken)

  VulkanDownloadToken(ObjectPtr<IVulkanTemporalBufferLease> lease, uint64_t size);
  VulkanDownloadToken(ObjectPtr<IVulkanTemporalBufferLease> lease, const DownloadLayout &layout);
  ~VulkanDownloadToken() override;

 protected:
  bool CopyData(void *destination) const override;

 private:
  ObjectPtr<IVulkanTemporalBufferLease> lease_;
  DownloadLayout layout_;
};
}  // namespace grassland::rhi::backend::vulkan
