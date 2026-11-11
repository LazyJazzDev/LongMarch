#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend::vulkan {
class IVulkanDownloadToken;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::vulkan::IVulkanDownloadToken)
#include "grassland/rhi/backend/backend_download_token.h"
#include "grassland/rhi/backend/vulkan/vulkan_temporal_buffer_pool.h"

namespace grassland::rhi::backend::vulkan {
class IVulkanDownloadToken : public DownloadToken {
 public:
  static IVulkanDownloadToken *Create(ObjectPtr<IVulkanTemporalBufferLease> lease, uint64_t size);
  static IVulkanDownloadToken *Create(ObjectPtr<IVulkanTemporalBufferLease> lease, const DownloadLayout &layout);
};

;
}  // namespace grassland::rhi::backend::vulkan
