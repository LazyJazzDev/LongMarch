#include "grassland/rhi/backend/backend_submission_token.h"

namespace grassland::rhi::backend {

SubmissionToken::SubmissionToken(Device *device, ObjectPtr<Event> event) : device_(device), event_(std::move(event)) {
}

IEvent *SubmissionToken::GetEvent() const {
  return ObjectPtr<Event>::Retain(event_.Get()).Detach();
}

Device *SubmissionToken::Owner() const {
  return device_;
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
