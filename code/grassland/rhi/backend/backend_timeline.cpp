#include "grassland/rhi/backend/backend_timeline.h"

#include "grassland/rhi/backend/backend_timeline_state.h"

LONGMARCH_RHI_UUID(grassland::rhi::backend::IBackendTimeline, "1eefb6fe-e220-483b-a0da-92f046be5fe6")

namespace grassland::rhi::backend {
void IBackendTimeline::InitializeTimeline(IBackendDevice *device) {
  StateTimeline().device_ = device;
}
}  // namespace grassland::rhi::backend

namespace grassland::rhi::backend {
TimelineState::TimelineState() = default;
TimelineState::~TimelineState() = default;
}  // namespace grassland::rhi::backend

namespace grassland::rhi::backend {
IBackendDevice *IBackendTimeline::Owner() const {
  return StateTimeline().device_;
}
}  // namespace grassland::rhi::backend
