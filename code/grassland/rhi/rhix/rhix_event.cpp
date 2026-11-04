#include "grassland/rhi/rhix/rhix_event.h"

namespace grassland::rhix {
Event::Event() noexcept = default;

Event::Event(rhi::IEvent *event) noexcept : ObjectContainer(event) {
}

rhi::IEvent *Event::Get() const noexcept {
  return static_cast<rhi::IEvent *>(GetObject());
}

rhi::SubmissionStatus Event::GetStatus() const {
  return Get() ? Get()->GetStatus() : rhi::SUBMISSION_STATUS_CANCELED;
}

bool Event::WaitHost() const {
  return Get() && Get()->WaitHost();
}
}  // namespace grassland::rhix
