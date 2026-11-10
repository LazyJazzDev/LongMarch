#pragma once
#include "grassland/rhi/rhi_timeline.h"

namespace grassland::rhi::backend {
class Device;

class Timeline : public ITimeline {
 public:
  explicit Timeline(Device *device) : device_(device) {
  }

  ~Timeline() override = default;

  Device *Owner() const {
    return device_;
  }

 private:
  Device *device_;
};

}  // namespace grassland::rhi::backend
