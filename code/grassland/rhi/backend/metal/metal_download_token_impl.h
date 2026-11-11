#pragma once
#include "grassland/rhi/backend/backend_download_token_state.h"
#include "grassland/rhi/backend/metal/metal_download_token.h"
#include "grassland/rhi/backend/metal/metal_util.h"

namespace grassland::rhi::backend::metal {
class MetalDownloadTokenImpl final : public Object<IMetalDownloadToken> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(IDownloadToken, IMetalDownloadToken, IBackendDownloadToken, IDownloadToken)
  MetalDownloadTokenImpl(ObjectPtr<IMetalTemporalBufferLease> lease, uint64_t size);
  MetalDownloadTokenImpl(ObjectPtr<IMetalTemporalBufferLease> lease, const DownloadLayout &layout);
  ~MetalDownloadTokenImpl() override;

 protected:
  bool CopyData(void *destination) const override;

 private:
  ObjectPtr<IMetalTemporalBufferLease> lease_;
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
}  // namespace grassland::rhi::backend::metal
