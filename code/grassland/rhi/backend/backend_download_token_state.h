#pragma once
#include "grassland/rhi/backend/backend_download_token.h"

namespace grassland::rhi::backend {
struct DownloadTokenState final {
  DownloadTokenState();
  ~DownloadTokenState();
  uint64_t size_{};
  ObjectPtr<SubmissionToken> submission_{};
};
}  // namespace grassland::rhi::backend
