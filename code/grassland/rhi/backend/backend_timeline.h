#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_timeline.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::IBackendTimeline)

namespace grassland::rhi::backend {
struct TimelineState;

class IBackendTimeline : public ITimeline {
 protected:
  void InitializeTimeline(IBackendDevice *device);

 public:
  IBackendDevice *Owner() const;

 protected:
  virtual TimelineState &StateTimeline() = 0;
  virtual const TimelineState &StateTimeline() const = 0;
};

}  // namespace grassland::rhi::backend
