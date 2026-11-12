#include "grassland/rhi/backend/metal/metal_timeline.h"

namespace grassland::rhi::backend::metal {
ObjectPtr<IMetalTimeline> IMetalTimeline::Create(IMetalDevice *device) {
  return MetalTimelineImpl::Create(device);
}
}  // namespace grassland::rhi::backend::metal
