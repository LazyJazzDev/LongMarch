#include "grassland/rhi/backend/backend_event.h"

#include "grassland/rhi/backend/backend_device.h"
#include "grassland/rhi/backend/backend_queue.h"

namespace grassland::rhi::backend {
Event::Event(Queue *queue, uint64_t value) : point_{queue, value} {
}

SubmissionStatus Event::GetStatus() const {
  return Owner()->QueryEvent(point_);
}

bool Event::WaitHost() const {
  return Owner()->WaitTimelines({point_}, true);
}

Device *Event::Owner() const {
  return point_.queue->Owner();
}

TimelinePoint Event::Point() const {
  return point_;
}

}  // namespace grassland::rhi::backend
