#pragma once
#include "grassland/rhi/backend/metal/metal_download_token.h"
#include "grassland/rhi/backend/metal/metal_util.h"

namespace grassland::rhi::backend::metal {
class MetalDownloadTokenImpl final : public Object<rhi::IDownloadToken, IBackendDownloadToken> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(IDownloadToken, IBackendDownloadToken, IDownloadToken)

  MetalDownloadTokenImpl(ObjectPtr<IMetalTemporalBufferLease> lease, uint64_t size);

  MetalDownloadTokenImpl(ObjectPtr<IMetalTemporalBufferLease> lease, const DownloadLayout &layout);

  ~MetalDownloadTokenImpl() override;

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

  ObjectPtr<IMetalTemporalBufferLease> lease_;

  DownloadLayout layout_;

  void InitializeDownloadToken(uint64_t size) {
    size_ = size;
  }

  uint64_t size_{};

  ObjectPtr<IBackendSubmissionToken> submission_{};
};
}  // namespace grassland::rhi::backend::metal
