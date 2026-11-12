#pragma once
#include "grassland/rhi/backend/backend_submission_token.h"
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_download_token.h"

namespace grassland::rhi::backend {
// Describes the CPU copy from a native readback allocation. A nonzero stencil
// pitch denotes a separate stencil plane, packed into the fourth output byte.
struct DownloadLayout {
  uint64_t offset{}, row_pitch{}, row_bytes{};
  uint32_t rows{1};
  uint64_t stencil_offset{}, stencil_pitch{};
};

}  // namespace grassland::rhi::backend
