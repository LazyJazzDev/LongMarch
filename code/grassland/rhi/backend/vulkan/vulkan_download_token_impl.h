#pragma once
#include "grassland/rhi/backend/vulkan/vulkan_download_token.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {
class VulkanDownloadTokenImpl final : public Object<rhi::IDownloadToken, IBackendDownloadToken> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(IDownloadToken, IBackendDownloadToken, IDownloadToken)

  VulkanDownloadTokenImpl(ObjectPtr<IVulkanTemporalBufferLease> lease, uint64_t size);

  VulkanDownloadTokenImpl(ObjectPtr<IVulkanTemporalBufferLease> lease, const DownloadLayout &layout);

  ~VulkanDownloadTokenImpl() override;

  SubmissionStatus GetStatus() const override {
    return submission_ ? rhi::QueryInterfacePtr<ISubmissionToken>(submission_.Get())->GetStatus()
                       : SUBMISSION_STATUS_PENDING;
  }

  bool WaitHost() const override {
    return submission_ && rhi::QueryInterfacePtr<ISubmissionToken>(submission_.Get())->WaitHost();
  }

  uint64_t GetSize() const override {
    return size_;
  }

  bool CopyTo(void *destination, uint64_t size) const override {
    if (!destination || size < size_)
      return false;
    return CopyData(destination);
  }

  void BindSubmission(IBackendSubmissionToken *submission) override {
    submission_ = ObjectPtr<IBackendSubmissionToken>::Retain(submission);
  }

 private:
  bool CopyData(void *destination) const override;

  ObjectPtr<IVulkanTemporalBufferLease> lease_;

  DownloadLayout layout_;

  void InitializeDownloadToken(uint64_t size) {
    size_ = size;
  }

  uint64_t size_{};

  ObjectPtr<IBackendSubmissionToken> submission_{};
};
}  // namespace grassland::rhi::backend::vulkan
