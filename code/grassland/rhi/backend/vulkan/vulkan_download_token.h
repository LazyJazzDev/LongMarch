#pragma once
#include "grassland/rhi/backend/backend_download_token.h"
#include "grassland/rhi/backend/vulkan/vulkan_temporal_buffer_pool.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::vulkan::IVulkanDownloadToken)

namespace grassland::rhi::backend::vulkan {
class IVulkanDownloadToken : public IBackendDownloadToken {
 public:
  static IVulkanDownloadToken *Create(ObjectPtr<IVulkanTemporalBufferLease> lease, uint64_t size);
  static IVulkanDownloadToken *Create(ObjectPtr<IVulkanTemporalBufferLease> lease, const DownloadLayout &layout);
};

}  // namespace grassland::rhi::backend::vulkan
