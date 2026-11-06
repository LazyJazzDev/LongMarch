#include "grassland/rhi/backend/backend_timeline.h"

#include <algorithm>

#include "grassland/rhi/backend/backend_device.h"

namespace grassland::rhi::backend {
Timeline::Timeline(Device *device) : device_(device) {
}

Device *Timeline::Owner() const {
  return device_;
}

bool Timeline::GetCompletedValue(uint64_t &value) const {
  value = completed_.load();
  return true;
}

bool Timeline::WaitHost(uint64_t value) const {
  return device_->WaitTimelines({{const_cast<Timeline *>(this), value}}, true);
}

bool Timeline::IsNative() const {
  return false;
}

void Timeline::RecordCompletion(uint64_t value) {
  completed_.store(value);
}

std::vector<Event> CoalesceEvents(const std::vector<Event> &events, bool wait_all) {
  std::vector<Event> result;
  for (auto event : events) {
    auto existing =
        std::find_if(result.begin(), result.end(), [&](const Event &p) { return p.timeline == event.timeline; });
    if (existing == result.end())
      result.push_back(event);
    else
      existing->value = wait_all ? std::max(existing->value, event.value) : std::min(existing->value, event.value);
  }
  return result;
}
}  // namespace grassland::rhi::backend
