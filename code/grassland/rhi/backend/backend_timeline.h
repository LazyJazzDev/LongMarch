#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_timeline.h"

namespace grassland::rhi::backend {
class Device;

class Timeline : public Object<ITimeline> {
 public:
  LONGMARCH_RHI_QUERY(Timeline, Object<ITimeline>)

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
