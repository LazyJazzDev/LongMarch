#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_event.h"

namespace grassland::rhi::backend {
class Device;
class Queue;

struct TimelinePoint {
  Queue *queue;
  uint64_t value;
};

class Event final : public Object<IEvent> {
 public:
  Event(Queue *queue, uint64_t value);
  SubmissionStatus GetStatus() const override;
  bool WaitHost() const override;
  Device *Owner() const;
  TimelinePoint Point() const;

 private:
  const TimelinePoint point_;
};
}  // namespace grassland::rhi::backend
