#pragma once
#include "grassland/rhi/backend/d3d12/d3d12_download_token.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12DownloadTokenImpl final : public Object<ID3D12DownloadToken> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(IDownloadToken, ID3D12DownloadToken, IBackendDownloadToken, IDownloadToken)

  D3D12DownloadTokenImpl(ObjectPtr<ID3D12TemporalBufferLease> lease, uint64_t size);

  D3D12DownloadTokenImpl(ObjectPtr<ID3D12TemporalBufferLease> lease, const DownloadLayout &layout);

  ~D3D12DownloadTokenImpl() override;

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

  void BindSubmission(IBackendSubmissionToken *submission) override {
    submission_ = ObjectPtr<IBackendSubmissionToken>::Retain(submission);
  }

 private:
  bool CopyData(void *destination) const override;

  ObjectPtr<ID3D12TemporalBufferLease> lease_;

  DownloadLayout layout_;

  void InitializeDownloadToken(uint64_t size) {
    size_ = size;
  }

  uint64_t size_{};

  ObjectPtr<IBackendSubmissionToken> submission_{};
};
}  // namespace grassland::rhi::backend::d3d12
