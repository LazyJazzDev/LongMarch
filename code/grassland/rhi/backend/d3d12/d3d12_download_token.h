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
