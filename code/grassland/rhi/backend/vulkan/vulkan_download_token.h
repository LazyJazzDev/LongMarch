#pragma once
#include "grassland/rhi/backend/backend_download_token.h"
#include "grassland/rhi/backend/vulkan/vulkan_temporal_buffer_pool.h"

namespace grassland::rhi::backend::vulkan {
class VulkanDownloadToken final : public DownloadToken {
 public:
  void AddRef() noexcept override {
    RetainReference();
  }

  void Release() noexcept override {
    if (ReleaseReference())
      delete this;
  }

  SubmissionStatus GetStatus() const override {
    return DownloadStatus();
  }

  bool WaitHost() const override {
    return WaitForDownload();
  }

  uint64_t GetSize() const override {
    return DownloadSize();
  }

  bool CopyTo(void *destination, uint64_t size) const override {
    return CopyDownload(destination, size);
  }

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
