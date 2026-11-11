#pragma once
#include "grassland/rhi/backend/backend_download_token.h"
#include "grassland/rhi/backend/backend_util.h"

namespace grassland::rhi::backend {
struct DownloadTokenState final {
  DownloadTokenState();
  ~DownloadTokenState();
  uint64_t size_{};
  ObjectPtr<IBackendSubmissionToken> submission_{};
};
}  // namespace grassland::rhi::backend
