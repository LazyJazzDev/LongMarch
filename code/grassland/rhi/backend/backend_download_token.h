#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend {
class DownloadToken;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::DownloadToken)

#include "grassland/rhi/backend/backend_submission_token.h"
#include "grassland/rhi/rhi_download_token.h"

namespace grassland::rhi::backend {
// Describes the CPU copy from a native readback allocation. A nonzero stencil
// pitch denotes a separate stencil plane, packed into the fourth output byte.
struct DownloadLayout {
  uint64_t offset{}, row_pitch{}, row_bytes{};
  uint32_t rows{1};
  uint64_t stencil_offset{}, stencil_pitch{};
};

struct DownloadTokenState;

class DownloadToken : public IDownloadToken {
 protected:
  void InitializeDownloadToken(uint64_t size);
  void DestroyDownloadToken();

 public:
  SubmissionStatus GetStatus() const override;
  bool WaitHost() const override;
  uint64_t GetSize() const override;
  bool CopyTo(void *destination, uint64_t size) const override;
  void BindSubmission(SubmissionToken *submission);

 protected:
  virtual bool CopyData(void *destination) const = 0;
  virtual DownloadTokenState &StateDownloadToken() = 0;
  virtual const DownloadTokenState &StateDownloadToken() const = 0;
};
}  // namespace grassland::rhi::backend
