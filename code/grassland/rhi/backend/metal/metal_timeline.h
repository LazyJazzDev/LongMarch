#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend::metal {
class MetalTimeline;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::metal::MetalTimeline)
#include <Metal/Metal.hpp>
#include <functional>
#include <memory>

#include "grassland/rhi/backend/backend_timeline.h"

namespace grassland::rhi::backend::metal {
class MetalDevice;

class MetalTimeline final : public Timeline {
 public:
  LONGMARCH_RHI_QUERY(MetalTimeline, Timeline)

  static ObjectPtr<MetalTimeline> Create(MetalDevice *device);
  bool GetCompletedValue(uint64_t &value) const override;
  bool WaitHost(uint64_t value) const override;
  MTL::Event *NativeEvent() const;
  MTL::SharedEvent *HostEvent() const;
  void MarkFailed();
  void ObserveCompletion(MTL::CommandBuffer *commands, uint64_t value, std::function<void()> notify);

 private:
  explicit MetalTimeline(MetalDevice *device);
  struct State;
  std::shared_ptr<State> state_;
  NS::SharedPtr<MTL::Event> gpu_event_;
  NS::SharedPtr<MTL::SharedEvent> event_;
};

}  // namespace grassland::rhi::backend::metal
