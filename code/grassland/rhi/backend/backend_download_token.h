#pragma once
#include "grassland/rhi/backend/backend_submission_token.h"
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_download_token.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::IBackendDownloadToken,
                        grassland::rhi::IObject,
                        "ed7d2ceb-b722-4881-ad53-9d19894b4bb2")

namespace grassland::rhi::backend {
// Describes the CPU copy from a native readback allocation. A nonzero stencil
// pitch denotes a separate stencil plane, packed into the fourth output byte.
struct DownloadLayout {
  uint64_t offset{}, row_pitch{}, row_bytes{};
  uint32_t rows{1};
  uint64_t stencil_offset{}, stencil_pitch{};
};

class IBackendDownloadToken : public IObject {
 public:
  virtual void BindSubmission(rhi::ISubmissionToken *submission) = 0;

 protected:
  virtual bool CopyData(void *destination) const = 0;
};
}  // namespace grassland::rhi::backend
