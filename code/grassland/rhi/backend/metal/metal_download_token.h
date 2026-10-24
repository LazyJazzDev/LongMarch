#pragma once
#include "grassland/rhi/backend/backend_download_token.h"
#include "grassland/rhi/backend/metal/metal_temporal_buffer_pool.h"

namespace grassland::rhi::backend::metal {
class MetalDownloadToken final : public DownloadToken {
 public:
  MetalDownloadToken(ObjectPtr<MetalTemporalBufferLease> lease, uint64_t size);
  MetalDownloadToken(ObjectPtr<MetalTemporalBufferLease> lease, const DownloadLayout &layout);
  ~MetalDownloadToken() override;

 protected:
  bool CopyData(void *destination) const override;

 private:
  ObjectPtr<MetalTemporalBufferLease> lease_;
  DownloadLayout layout_;
};
}  // namespace grassland::rhi::backend::metal
