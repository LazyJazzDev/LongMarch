#pragma once
#include "grassland/rhi/backend/backend_download_token_state.h"
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/backend/vulkan/vulkan_download_token.h"

namespace grassland::rhi::backend::vulkan {
class VulkanDownloadTokenImpl final : public Object<IVulkanDownloadToken> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(IDownloadToken, IVulkanDownloadToken, IBackendDownloadToken, IDownloadToken)
  VulkanDownloadTokenImpl(ObjectPtr<IVulkanTemporalBufferLease> lease, uint64_t size);
  VulkanDownloadTokenImpl(ObjectPtr<IVulkanTemporalBufferLease> lease, const DownloadLayout &layout);
  ~VulkanDownloadTokenImpl() override;

 protected:
  bool CopyData(void *destination) const override;

 private:
  ObjectPtr<IVulkanTemporalBufferLease> lease_;
  DownloadLayout layout_;

 private:
  DownloadTokenState state_DownloadToken_;

  DownloadTokenState &StateDownloadToken() override {
    return state_DownloadToken_;
  }

  const DownloadTokenState &StateDownloadToken() const override {
    return state_DownloadToken_;
  }

 public:
  SubmissionStatus GetStatus() const override {
    return IBackendDownloadToken::CommonGetStatus();
  }

  bool WaitHost() const override {
    return IBackendDownloadToken::CommonWaitHost();
  }

  uint64_t GetSize() const override {
    return IBackendDownloadToken::CommonGetSize();
  }

  bool CopyTo(void *destination, uint64_t size) const override {
    return IBackendDownloadToken::CommonCopyTo(destination, size);
  }
};
}  // namespace grassland::rhi::backend::vulkan
