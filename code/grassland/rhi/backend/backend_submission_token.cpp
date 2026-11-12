#include "grassland/rhi/backend/backend_submission_token.h"

namespace grassland::rhi::backend {

SubmissionTokenImpl::SubmissionTokenImpl(Event event) : event_(event) {
}

Event SubmissionTokenImpl::GetEvent() const {
  return event_;
}

SubmissionStatus SubmissionTokenImpl::GetStatus() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return status_;
}

bool SubmissionTokenImpl::WaitHost() const {
  std::unique_lock<std::mutex> lock(mutex_);
  changed_.wait(lock, [&] { return status_ != SUBMISSION_STATUS_PENDING; });
  return status_ == SUBMISSION_STATUS_COMPLETED;
}

void SubmissionTokenImpl::Complete(SubmissionStatus status) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    status_ = status;
  }
  changed_.notify_all();
}

IBackendSubmissionToken *IBackendSubmissionToken::Create(Event event) {
  return new SubmissionTokenImpl(event);
}
}  // namespace grassland::rhi::backend
