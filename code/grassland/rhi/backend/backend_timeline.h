#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend {
class Timeline;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::Timeline)
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_timeline.h"

namespace grassland::rhi::backend {
class Device;

struct TimelineState;

class Timeline : public ITimeline {
 protected:
  void InitializeTimeline(Device *device);

 public:
  Device *Owner() const;

 protected:
  virtual TimelineState &StateTimeline() = 0;
  virtual const TimelineState &StateTimeline() const = 0;
};

}  // namespace grassland::rhi::backend
