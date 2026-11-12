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
