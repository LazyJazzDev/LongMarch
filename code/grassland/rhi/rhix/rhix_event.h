#pragma once
#include "grassland/rhi/rhi_event.h"
#include "grassland/rhi/rhix/rhix_object_container.h"

namespace grassland::rhix {
class Event : public ObjectContainer {
 public:
  Event() noexcept;
  Event(rhi::IEvent *event) noexcept;
  rhi::IEvent *Get() const noexcept;
  rhi::SubmissionStatus GetStatus() const;
  bool WaitHost() const;
};
}  // namespace grassland::rhix
