#pragma once
#include <cstdint>

namespace grassland::rhi {

// Read-only view of a device-owned GPU timeline. Neither this view nor an Event
// retains the device; callers must finish using them before releasing it.
class ITimeline {
 public:
  virtual bool GetCompletedValue(uint64_t &value) const = 0;
  virtual bool WaitHost(uint64_t value) const = 0;

 protected:
  virtual ~ITimeline() = default;
};

struct Event {
  ITimeline *timeline{};
  uint64_t value{};
};

}  // namespace grassland::rhi
