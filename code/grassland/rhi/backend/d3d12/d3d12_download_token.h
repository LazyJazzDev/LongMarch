#pragma once
#include "grassland/rhi/backend/backend_download_token.h"
#include "grassland/rhi/backend/d3d12/d3d12_temporal_buffer_pool.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

namespace grassland::rhi::backend::d3d12 {
IBackendDownloadToken *CreateD3D12DownloadToken(ObjectPtr<ID3D12TemporalBufferLease> lease, uint64_t size);
IBackendDownloadToken *CreateD3D12DownloadToken(ObjectPtr<ID3D12TemporalBufferLease> lease,
                                                const DownloadLayout &layout);

}  // namespace grassland::rhi::backend::d3d12
