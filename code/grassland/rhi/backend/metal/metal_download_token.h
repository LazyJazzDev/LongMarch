#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend::metal {
class MetalDownloadToken;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::metal::MetalDownloadToken)
#include "grassland/rhi/backend/backend_download_token.h"
#include "grassland/rhi/backend/metal/metal_temporal_buffer_pool.h"

namespace grassland::rhi::backend::metal {
class MetalDownloadToken final : public DownloadToken {
 public:
  LONGMARCH_RHI_QUERY(MetalDownloadToken, DownloadToken)

  MetalDownloadToken(ObjectPtr<IMetalTemporalBufferLease> lease, uint64_t size);
  MetalDownloadToken(ObjectPtr<IMetalTemporalBufferLease> lease, const DownloadLayout &layout);
  ~MetalDownloadToken() override;

 protected:
  bool CopyData(void *destination) const override;

 private:
  ObjectPtr<IMetalTemporalBufferLease> lease_;
  DownloadLayout layout_;
};
}  // namespace grassland::rhi::backend::metal
