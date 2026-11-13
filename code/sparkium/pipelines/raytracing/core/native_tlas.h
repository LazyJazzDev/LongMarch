#pragma once

#include <memory>

#include "grassland/rhi/rhi_device.h"
#include "sparkium/core/resource_updates.h"

namespace sparkium::raytracing {

// One ordered rendering stream. Like RHI AS recording, a modifying context must
// be submitted or discarded before preparing another context for this object.
class NativeTLAS {
 public:
  bool Record(grassland::rhi::IDevice *device,
              grassland::rhi::ICommandContext *commands,
              ResourceUpdates::Cursor &updates,
              const std::vector<grassland::rhi::RayTracingInstance> &instances);
  grassland::rhi::ITopLevelAccelerationStructure *Get() const;

 private:
  struct State {
    grassland::rhi::ObjPtr<grassland::rhi::ITopLevelAccelerationStructure> structure;
    std::vector<grassland::rhi::RayTracingInstance> instances;
  };

  grassland::rhi::ObjPtr<grassland::rhi::ITopLevelAccelerationStructure> structure_;
  std::shared_ptr<State> accepted_ = std::make_shared<State>();
};

}  // namespace sparkium::raytracing
