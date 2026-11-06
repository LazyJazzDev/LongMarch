#pragma once
#include "grassland/rhi/rhi_object.h"

namespace grassland::rhi {
// Read-only synchronization interface. Producers control signaling.
class ITimeline : public IObject {
 public:
  // Returns false on device failure; value is valid only on success.
  virtual bool GetCompletedValue(uint64_t &value) const = 0;
  virtual bool WaitHost(uint64_t value) const = 0;

 protected:
  ~ITimeline() override;
};
}  // namespace grassland::rhi
