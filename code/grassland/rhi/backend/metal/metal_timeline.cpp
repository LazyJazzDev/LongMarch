#include "grassland/rhi/backend/metal/metal_timeline.h"

#include "grassland/rhi/backend/metal/metal_device.h"

namespace grassland::rhi::backend::metal {
MetalTimeline::MetalTimeline(MetalDevice *device) : backend::Timeline(device) {
}

MetalTimeline *MetalTimeline::Create(MetalDevice *device) {
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto result = std::unique_ptr<MetalTimeline>(new MetalTimeline(device));
  result->event_ = NS::TransferPtr(device->NativeDevice()->newSharedEvent());
  if (!result->event_) {
    ReportError(-1, "Create Metal timeline failed.");
    return nullptr;
  }
  return result.release();
}

MetalTimeline::~MetalTimeline() = default;

MTL::SharedEvent *MetalTimeline::Native() const {
  return event_.get();
}

bool MetalTimeline::GetCompletedValue(uint64_t &value) const {
  value = event_->signaledValue();
  return !failed_.load();
}

void MetalTimeline::RecordFailure() {
  failed_.store(true);
}

}  // namespace grassland::rhi::backend::metal
