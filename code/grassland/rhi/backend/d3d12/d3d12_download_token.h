#pragma once
#include "grassland/rhi/backend/backend_download_token.h"
#include "grassland/rhi/backend/d3d12/d3d12_temporal_buffer_pool.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

namespace grassland::rhi::backend::d3d12 {
class ID3D12DownloadToken : public IObject {
 public:
  virtual void BindSubmission(rhi::ISubmissionToken *submission) = 0;
};

ID3D12DownloadToken *CreateD3D12DownloadToken(ObjectPtr<ID3D12TemporalBufferLease> lease, uint64_t size);
ID3D12DownloadToken *CreateD3D12DownloadToken(ObjectPtr<ID3D12TemporalBufferLease> lease, const DownloadLayout &layout);

}  // namespace grassland::rhi::backend::d3d12

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::d3d12::ID3D12DownloadToken,
                        grassland::rhi::IObject,
                        "4124741e-664d-43ab-900f-06fd4b5698d6")

namespace grassland::rhi::backend::d3d12 {
class D3D12DownloadTokenImpl final : public Object<rhi::IDownloadToken, ID3D12DownloadToken> {
 public:
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

  void BindSubmission(rhi::ISubmissionToken *submission) override {
    submission_ = ObjectPtr<ISubmissionToken>::Retain(submission);
  }

 private:
  bool CopyData(void *destination) const;

  ObjectPtr<ID3D12TemporalBufferLease> lease_;

  DownloadLayout layout_;

  void InitializeDownloadToken(uint64_t size) {
    size_ = size;
  }

  uint64_t size_{};

  ObjectPtr<ISubmissionToken> submission_{};
};
}  // namespace grassland::rhi::backend::d3d12
