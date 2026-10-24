#include "grassland/rhi/backend/backend_download_token.h"

namespace grassland::rhi::backend {
DownloadToken::DownloadToken(uint64_t size) : size_(size) {
}

DownloadToken::~DownloadToken() = default;

SubmissionStatus DownloadToken::GetStatus() const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  return submission_ ? submission_->GetStatus() : unsubmitted_status_;
}

bool DownloadToken::WaitHost() const {
  ObjectPtr<SubmissionToken> submission;
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    submission = submission_;
  }
  // Waiting while still recording must not block the thread that submits work.
  return submission && submission->WaitHost();
}

uint64_t DownloadToken::GetSize() const {
  return size_;
}

bool DownloadToken::CopyTo(void *destination, uint64_t size) const {
  if (!destination || size < size_ || GetStatus() != SUBMISSION_STATUS_COMPLETED)
    return false;
  std::lock_guard<std::mutex> lock(copy_mutex_);
  return CopyData(destination);
}

void DownloadToken::BindSubmission(SubmissionToken *submission) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (!submission_ && unsubmitted_status_ == SUBMISSION_STATUS_PENDING)
    submission_ = ObjectPtr<SubmissionToken>::Retain(submission);
}

void DownloadToken::Cancel(SubmissionStatus status) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (!submission_)
    unsubmitted_status_ = status;
}
}  // namespace grassland::rhi::backend
