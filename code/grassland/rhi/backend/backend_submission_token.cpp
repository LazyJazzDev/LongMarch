#include "grassland/rhi/backend/backend_submission_token.h"

namespace grassland::rhi::backend {

SubmissionToken::SubmissionToken(uint64_t identity) : queue_identity_(identity) {
}

uint64_t SubmissionToken::QueueIdentity() const {
  return queue_identity_;
}

SubmissionStatus SubmissionToken::GetStatus() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return status_;
}

bool SubmissionToken::WaitHost() const {
  std::unique_lock<std::mutex> lock(mutex_);
  changed_.wait(lock, [&] { return status_ != SUBMISSION_STATUS_PENDING; });
  return status_ == SUBMISSION_STATUS_COMPLETED;
}

void SubmissionToken::Complete(SubmissionStatus status) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    status_ = status;
  }
  changed_.notify_all();
}

}  // namespace grassland::rhi::backend
