#pragma once
#include <Metal/Metal.hpp>
#include <functional>
#include <memory>

#include "grassland/rhi/backend/metal/metal_util.h"
#include "grassland/rhi/rhi_timeline.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::metal::IMetalTimeline,
                        grassland::rhi::IObject,
                        "ca12a3b9-c2c2-4f0b-b396-324edaa7e96c")

namespace grassland::rhi::backend::metal {
class IMetalTimeline : public IObject {
 public:
  virtual IMetalDevice *Owner() const = 0;
  static ObjectPtr<IMetalTimeline> Create(IMetalDevice *device);
  virtual MTL::Event *NativeEvent() const = 0;
  virtual MTL::SharedEvent *HostEvent() const = 0;
  virtual void MarkFailed() = 0;
  virtual void ObserveCompletion(MTL::CommandBuffer *commands, uint64_t value, std::function<void()> notify) = 0;
};

}  // namespace grassland::rhi::backend::metal
