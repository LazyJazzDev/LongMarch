#pragma once
#include <mutex>

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

class DownloadToken : public Object<IDownloadToken> {
 public:
  explicit DownloadToken(uint64_t size);
  ~DownloadToken() override;
  SubmissionStatus GetStatus() const override;
  bool WaitHost() const override;
  uint64_t GetSize() const override;
  bool CopyTo(void *destination, uint64_t size) const override;
  void BindSubmission(SubmissionToken *submission);
  void Cancel(SubmissionStatus status);

 protected:
  virtual bool CopyData(void *destination) const = 0;

 private:
  const uint64_t size_;
  mutable std::mutex state_mutex_;
  mutable std::mutex copy_mutex_;
  ObjectPtr<SubmissionToken> submission_;
  SubmissionStatus unsubmitted_status_{SUBMISSION_STATUS_PENDING};
};
}  // namespace grassland::rhi::backend
