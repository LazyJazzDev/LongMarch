#pragma once
#include <Metal/Metal.hpp>
#include <functional>
#include <memory>

#include "grassland/rhi/backend/backend_timeline.h"
#include "grassland/rhi/backend/metal/metal_util.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::metal::IMetalTimeline)

namespace grassland::rhi::backend::metal {
class IMetalTimeline : public IBackendTimeline {
 public:
  static ObjectPtr<IMetalTimeline> Create(IMetalDevice *device);
  virtual bool GetCompletedValue(uint64_t &value) const = 0;
  virtual bool WaitHost(uint64_t value) const = 0;
  virtual MTL::Event *NativeEvent() const = 0;
  virtual MTL::SharedEvent *HostEvent() const = 0;
  virtual void MarkFailed() = 0;
  virtual void ObserveCompletion(MTL::CommandBuffer *commands, uint64_t value, std::function<void()> notify) = 0;
};

;

}  // namespace grassland::rhi::backend::metal
