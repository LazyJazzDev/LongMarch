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

namespace grassland::rhi::backend::vulkan {
class VulkanDownloadTokenImpl final : public Object<rhi::IDownloadToken, IVulkanDownloadToken> {
 public:
  VulkanDownloadTokenImpl(ObjectPtr<IVulkanTemporalBufferLease> lease, uint64_t size);

  VulkanDownloadTokenImpl(ObjectPtr<IVulkanTemporalBufferLease> lease, const DownloadLayout &layout);

  ~VulkanDownloadTokenImpl() override;

  SubmissionStatus GetStatus() const override {
    return submission_ ? submission_->GetStatus() : SUBMISSION_STATUS_PENDING;
  }

  bool WaitHost() const override {
    return submission_ && submission_->WaitHost();
  }

  uint64_t GetSize() const override {
    return size_;
  }

  bool CopyTo(void *destination, uint64_t size) const override {
    if (!destination || size < size_)
      return false;
    return CopyData(destination);
  }

  void BindSubmission(rhi::ISubmissionToken *submission) override {
    submission_ = ObjectPtr<ISubmissionToken>::Retain(submission);
  }

 private:
  bool CopyData(void *destination) const;

  ObjectPtr<IVulkanTemporalBufferLease> lease_;

  DownloadLayout layout_;

  void InitializeDownloadToken(uint64_t size) {
    size_ = size;
  }

  uint64_t size_{};

  ObjectPtr<ISubmissionToken> submission_{};
};
}  // namespace grassland::rhi::backend::vulkan
