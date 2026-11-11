#include "grassland/rhi/backend/metal/metal_timeline.h"

#include "grassland/rhi/backend/metal/metal_timeline_impl.h"

namespace grassland::rhi::backend::metal {
ObjectPtr<IMetalTimeline> IMetalTimeline::Create(IMetalDevice *device) {
  return MetalTimelineImpl::Create(device);
}
}  // namespace grassland::rhi::backend::metal
