#pragma once
#include "grassland/rhi/backend/backend_download_token.h"
#include "grassland/rhi/backend/metal/metal_temporal_buffer_pool.h"
#include "grassland/rhi/backend/metal/metal_util.h"

namespace grassland::rhi::backend::metal {
IBackendDownloadToken *CreateMetalDownloadToken(ObjectPtr<IMetalTemporalBufferLease> lease, uint64_t size);
IBackendDownloadToken *CreateMetalDownloadToken(ObjectPtr<IMetalTemporalBufferLease> lease,
                                                const DownloadLayout &layout);

}  // namespace grassland::rhi::backend::metal
