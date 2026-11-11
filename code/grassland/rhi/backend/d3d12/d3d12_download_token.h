#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12DownloadToken;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::d3d12::D3D12DownloadToken)
#include "grassland/rhi/backend/backend_download_token.h"
#include "grassland/rhi/backend/d3d12/d3d12_temporal_buffer_pool.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12DownloadToken final : public DownloadToken {
 public:
  LONGMARCH_RHI_QUERY(D3D12DownloadToken, DownloadToken)

  D3D12DownloadToken(ObjectPtr<D3D12TemporalBufferLease> lease, uint64_t size);
  D3D12DownloadToken(ObjectPtr<D3D12TemporalBufferLease> lease, const DownloadLayout &layout);
  ~D3D12DownloadToken() override;

 protected:
  bool CopyData(void *destination) const override;

 private:
  ObjectPtr<D3D12TemporalBufferLease> lease_;
  DownloadLayout layout_;
};
}  // namespace grassland::rhi::backend::d3d12
