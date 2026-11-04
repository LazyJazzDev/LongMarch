#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_event.h"

namespace grassland::rhi::backend {
class Device;

struct TimelinePoint {
  QueueType queue;
  uint64_t value;
};

class Event final : public Object<IEvent> {
 public:
  Event(Device *device, QueueType queue);
  SubmissionStatus GetStatus() const override;
  bool WaitHost() const override;
  Device *Owner() const;
  TimelinePoint Point() const;
  // Bind exactly once, before publishing the accepted submission.
  void BindSignalValue(uint64_t value);

 private:
  Device *device_;
  TimelinePoint point_;
};
}  // namespace grassland::rhi::backend
