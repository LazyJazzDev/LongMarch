#include "grassland/rhi/rhi_event.h"

namespace grassland::rhi {
SubmissionStatus Event::GetStatus() const {
  if (!timeline)
    return SUBMISSION_STATUS_CANCELED;
  uint64_t completed{};
  if (!timeline->GetCompletedValue(completed))
    return SUBMISSION_STATUS_FAILED;
  return completed >= value ? SUBMISSION_STATUS_COMPLETED : SUBMISSION_STATUS_PENDING;
}

bool Event::WaitHost() const {
  return timeline && timeline->WaitHost(value);
}
}  // namespace grassland::rhi
