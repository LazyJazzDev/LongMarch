#include "grassland/rhi/backend/metal/metal_timeline.h"

#include "grassland/rhi/backend/metal/metal_timeline_impl.h"

namespace grassland::rhi::backend::metal {
ObjectPtr<IMetalTimeline> IMetalTimeline::Create(IMetalDevice *device) {
  return MetalTimelineImpl::Create(device);
}
}  // namespace grassland::rhi::backend::metal

LONGMARCH_RHI_UUID(grassland::rhi::backend::metal::IMetalTimeline, "ca12a3b9-c2c2-4f0b-b396-324edaa7e96c")
