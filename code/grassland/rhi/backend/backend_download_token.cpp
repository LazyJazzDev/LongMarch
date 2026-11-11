#include "grassland/rhi/backend/backend_download_token.h"

#include "grassland/rhi/backend/backend_download_token_state.h"

namespace grassland::rhi::backend {
void DownloadToken::InitializeDownloadToken(uint64_t size) {
  StateDownloadToken().size_ = size;
}

SubmissionStatus DownloadToken::GetStatus() const {
  return StateDownloadToken().submission_ ? StateDownloadToken().submission_->GetStatus() : SUBMISSION_STATUS_PENDING;
}

bool DownloadToken::WaitHost() const {
  return StateDownloadToken().submission_ && StateDownloadToken().submission_->WaitHost();
}

uint64_t DownloadToken::GetSize() const {
  return StateDownloadToken().size_;
}

bool DownloadToken::CopyTo(void *destination, uint64_t size) const {
  if (!destination || size < StateDownloadToken().size_)
    return false;
  return CopyData(destination);
}

void DownloadToken::BindSubmission(SubmissionToken *submission) {
  StateDownloadToken().submission_ = ObjectPtr<SubmissionToken>::Retain(submission);
}
}  // namespace grassland::rhi::backend

LONGMARCH_RHI_UUID(grassland::rhi::backend::DownloadToken, "ed7d2ceb-b722-4881-ad53-9d19894b4bb2")

namespace grassland::rhi::backend {
DownloadTokenState::DownloadTokenState() = default;
DownloadTokenState::~DownloadTokenState() = default;
}  // namespace grassland::rhi::backend
