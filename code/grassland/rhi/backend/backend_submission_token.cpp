#include "grassland/rhi/backend/backend_submission_token.h"

#include "grassland/rhi/backend/backend_submission_token_impl.h"

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

}  // namespace grassland::rhi::backend

LONGMARCH_RHI_UUID(grassland::rhi::backend::IBackendSubmissionToken, "92c56d21-d1e3-4070-a232-2cc80606c437")

namespace grassland::rhi::backend {
IBackendSubmissionToken *IBackendSubmissionToken::Create(Event event) {
  return new SubmissionTokenImpl(event);
}
}  // namespace grassland::rhi::backend
