#pragma once
#include "grassland/rhi/backend/backend_download_token.h"
#include "grassland/rhi/backend/metal/metal_temporal_buffer_pool.h"
#include "grassland/rhi/backend/metal/metal_util.h"

namespace grassland::rhi::backend::metal {
class IMetalDownloadToken : public IObject {
 public:
  virtual void BindSubmission(rhi::ISubmissionToken *submission) = 0;
};

IMetalDownloadToken *CreateMetalDownloadToken(ObjectPtr<IMetalTemporalBufferLease> lease, uint64_t size);
IMetalDownloadToken *CreateMetalDownloadToken(ObjectPtr<IMetalTemporalBufferLease> lease, const DownloadLayout &layout);

}  // namespace grassland::rhi::backend::metal

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::metal::IMetalDownloadToken,
                        grassland::rhi::IObject,
                        "1c2e1359-b965-4369-9a13-f2d9861a0a9a")

namespace grassland::rhi::backend::metal {
class MetalDownloadTokenImpl final : public Object<rhi::IDownloadToken, IMetalDownloadToken> {
 public:
  MetalDownloadTokenImpl(ObjectPtr<IMetalTemporalBufferLease> lease, uint64_t size);

  MetalDownloadTokenImpl(ObjectPtr<IMetalTemporalBufferLease> lease, const DownloadLayout &layout);

  ~MetalDownloadTokenImpl() override;

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

  ObjectPtr<IMetalTemporalBufferLease> lease_;

  DownloadLayout layout_;

  void InitializeDownloadToken(uint64_t size) {
    size_ = size;
  }

  uint64_t size_{};

  ObjectPtr<ISubmissionToken> submission_{};
};
}  // namespace grassland::rhi::backend::metal
