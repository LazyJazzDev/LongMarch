#include "grassland/rhi/backend/backend_download_token.h"

namespace grassland::rhi::backend {
DownloadToken::DownloadToken(uint64_t size) : size_(size) {
}

DownloadToken::~DownloadToken() = default;

SubmissionStatus DownloadToken::GetStatus() const {
  return submission_ ? submission_->GetStatus() : SUBMISSION_STATUS_PENDING;
}

bool DownloadToken::WaitHost() const {
  return submission_ && submission_->WaitHost();
}

uint64_t DownloadToken::GetSize() const {
  return size_;
}

bool DownloadToken::CopyTo(void *destination, uint64_t size) const {
  if (!destination || size < size_)
    return false;
  return CopyData(destination);
}

void DownloadToken::BindSubmission(SubmissionToken *submission) {
  submission_ = ObjectPtr<SubmissionToken>::Retain(submission);
}
}  // namespace grassland::rhi::backend
