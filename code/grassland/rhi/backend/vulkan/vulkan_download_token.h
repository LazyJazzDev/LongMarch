#pragma once
#include "grassland/rhi/backend/backend_download_token.h"
#include "grassland/rhi/backend/vulkan/vulkan_temporal_buffer_pool.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {
class IVulkanDownloadToken : public IObject {
 public:
  virtual void BindSubmission(rhi::ISubmissionToken *submission) = 0;
};

IVulkanDownloadToken *CreateVulkanDownloadToken(ObjectPtr<IVulkanTemporalBufferLease> lease, uint64_t size);
IVulkanDownloadToken *CreateVulkanDownloadToken(ObjectPtr<IVulkanTemporalBufferLease> lease,
                                                const DownloadLayout &layout);

}  // namespace grassland::rhi::backend::vulkan

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::vulkan::IVulkanDownloadToken,
                        grassland::rhi::IObject,
                        "3f66ba7b-d5cc-45bb-a829-d4840a94feb1")
