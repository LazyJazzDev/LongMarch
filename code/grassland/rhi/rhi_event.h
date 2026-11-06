#pragma once
#include "grassland/rhi/rhi_timeline.h"

namespace grassland::rhi {
// A borrowed timeline and target value. Copies retain nothing. The timeline and
// its Device must outlive every use of this event, including submitted waits.
struct Event {
  ITimeline *timeline{};
  uint64_t value{};

  explicit operator bool() const {
    return timeline != nullptr;
  }

  SubmissionStatus GetStatus() const;
  bool WaitHost() const;
};
}  // namespace grassland::rhi
