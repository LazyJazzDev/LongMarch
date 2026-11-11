#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_interface_id.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::metal::IMetalDownloadToken)
#include "grassland/rhi/backend/backend_download_token.h"
#include "grassland/rhi/backend/metal/metal_temporal_buffer_pool.h"

namespace grassland::rhi::backend::metal {
class IMetalDownloadToken : public IBackendDownloadToken {
 public:
  static IMetalDownloadToken *Create(ObjectPtr<IMetalTemporalBufferLease> lease, uint64_t size);
  static IMetalDownloadToken *Create(ObjectPtr<IMetalTemporalBufferLease> lease, const DownloadLayout &layout);
};

;
}  // namespace grassland::rhi::backend::metal
