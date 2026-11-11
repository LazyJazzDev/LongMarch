#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend::d3d12 {
class ID3D12DownloadToken;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::d3d12::ID3D12DownloadToken)
#include "grassland/rhi/backend/backend_download_token.h"
#include "grassland/rhi/backend/d3d12/d3d12_temporal_buffer_pool.h"

namespace grassland::rhi::backend::d3d12 {
class ID3D12DownloadToken : public DownloadToken {
 public:
  static ID3D12DownloadToken *Create(ObjectPtr<ID3D12TemporalBufferLease> lease, uint64_t size);
  static ID3D12DownloadToken *Create(ObjectPtr<ID3D12TemporalBufferLease> lease, const DownloadLayout &layout);
};

;
}  // namespace grassland::rhi::backend::d3d12
