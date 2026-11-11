#pragma once
#include "grassland/rhi/backend/backend_download_token.h"
#include "grassland/rhi/backend/vulkan/vulkan_temporal_buffer_pool.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {
IBackendDownloadToken *CreateVulkanDownloadToken(ObjectPtr<IVulkanTemporalBufferLease> lease, uint64_t size);
IBackendDownloadToken *CreateVulkanDownloadToken(ObjectPtr<IVulkanTemporalBufferLease> lease,
                                                 const DownloadLayout &layout);

}  // namespace grassland::rhi::backend::vulkan
