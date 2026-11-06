#pragma once
#include <atomic>

#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_event.h"

namespace grassland::rhi::backend {
class Device;

class Timeline : public Object<ITimeline> {
 public:
  explicit Timeline(Device *device);
  Device *Owner() const;
  bool GetCompletedValue(uint64_t &value) const override;
  bool WaitHost(uint64_t value) const override;
  virtual bool IsNative() const;
  void RecordCompletion(uint64_t value);

 private:
  Device *device_;
  std::atomic<uint64_t> completed_{};
};

// Coalesce by actual timeline identity, not Queue roles.
std::vector<Event> CoalesceEvents(const std::vector<Event> &events, bool wait_all);
}  // namespace grassland::rhi::backend
