#pragma once
#include <Metal/Metal.hpp>

#include "grassland/rhi/backend/backend_timeline.h"

namespace grassland::rhi::backend::metal {
class MetalDevice;

class MetalTimeline final : public backend::Timeline {
 public:
  static MetalTimeline *Create(MetalDevice *device);
  ~MetalTimeline() override;
  MTL::SharedEvent *Native() const;
  bool GetCompletedValue(uint64_t &value) const override;

  bool IsNative() const override {
    return true;
  }

  void RecordFailure();

 private:
  explicit MetalTimeline(MetalDevice *device);
  NS::SharedPtr<MTL::SharedEvent> event_;
  std::atomic<bool> failed_{};
};
}  // namespace grassland::rhi::backend::metal
