#pragma once
#include <cstdint>

#include "grassland/rhi/rhi_object.h"

LONGMARCH_RHI_UUID(grassland::rhi::ITimeline, "92b5493b-5283-4f50-827f-6c5b4f5562a2")

namespace grassland::rhi {

// Borrowed read-only view of a device-owned GPU timeline. Neither this view nor an Event
// retains the device; callers must finish using them before releasing it.
// QueryInterface returns an owned interface reference, which must also be released first.
class ITimeline : public IObject {
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
