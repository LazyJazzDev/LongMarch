#pragma once
#include "grassland/rhi/backend/backend_download_token_state.h"
#include "grassland/rhi/backend/d3d12/d3d12_download_token.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12DownloadTokenImpl final : public Object<ID3D12DownloadToken> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(IDownloadToken, ID3D12DownloadToken, DownloadToken, IDownloadToken)
  D3D12DownloadTokenImpl(ObjectPtr<ID3D12TemporalBufferLease> lease, uint64_t size);
  D3D12DownloadTokenImpl(ObjectPtr<ID3D12TemporalBufferLease> lease, const DownloadLayout &layout);
  ~D3D12DownloadTokenImpl() override;

 protected:
  bool CopyData(void *destination) const override;

 private:
  ObjectPtr<ID3D12TemporalBufferLease> lease_;
  DownloadLayout layout_;

 private:
  DownloadTokenState state_DownloadToken_;

  DownloadTokenState &StateDownloadToken() override {
    return state_DownloadToken_;
  }

  const DownloadTokenState &StateDownloadToken() const override {
    return state_DownloadToken_;
  }
};
}  // namespace grassland::rhi::backend::d3d12
