#pragma once
#include "grassland/rhi/backend/backend_submission_token.h"
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_download_token.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::IBackendDownloadToken)

namespace grassland::rhi::backend {
// Describes the CPU copy from a native readback allocation. A nonzero stencil
// pitch denotes a separate stencil plane, packed into the fourth output byte.
struct DownloadLayout {
  uint64_t offset{}, row_pitch{}, row_bytes{};
  uint32_t rows{1};
  uint64_t stencil_offset{}, stencil_pitch{};
};

class IBackendDownloadToken : public IDownloadToken {
 public:
  SubmissionStatus GetStatus() const override = 0;

  bool WaitHost() const override = 0;

  uint64_t GetSize() const override = 0;

  bool CopyTo(void *destination, uint64_t size) const override = 0;

  virtual void BindSubmission(IBackendSubmissionToken *submission) = 0;

 protected:
  virtual bool CopyData(void *destination) const = 0;
};
}  // namespace grassland::rhi::backend
