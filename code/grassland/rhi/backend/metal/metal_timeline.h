#pragma once
#include <Metal/Metal.hpp>
#include <memory>

#include "grassland/rhi/backend/backend_timeline.h"

namespace grassland::rhi::backend::metal {
class MetalDevice;

class MetalTimeline final : public Timeline {
 public:
  static std::unique_ptr<MetalTimeline> Create(MetalDevice *device);
  bool GetCompletedValue(uint64_t &value) const override;
  bool WaitHost(uint64_t value) const override;
  MTL::SharedEvent *NativeEvent() const;
  void MarkFailed();

 private:
  explicit MetalTimeline(MetalDevice *device);
  struct State;
  std::shared_ptr<State> state_;
  NS::SharedPtr<MTL::SharedEvent> event_;
};

}  // namespace grassland::rhi::backend::metal
