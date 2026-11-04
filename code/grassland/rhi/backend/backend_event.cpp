#include "grassland/rhi/backend/backend_event.h"

#include <cassert>

#include "grassland/rhi/backend/backend_device.h"

namespace grassland::rhi::backend {
Event::Event(Device *device, QueueType queue) : device_(device), point_{queue, 0} {
}

SubmissionStatus Event::GetStatus() const {
  return device_->QueryEvent(point_);
}

bool Event::WaitHost() const {
  return device_->WaitTimelines({point_}, true);
}

Device *Event::Owner() const {
  return device_;
}

TimelinePoint Event::Point() const {
  return point_;
}

void Event::BindSignalValue(uint64_t value) {
  assert(point_.value == 0 && value != 0);
  point_.value = value;
}
}  // namespace grassland::rhi::backend
