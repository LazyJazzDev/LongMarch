#pragma once
#include <array>
#include <atomic>
#include <mutex>
#include <thread>

#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_acceleration_structure.h"
#include "grassland/rhi/rhi_device.h"
#include "grassland/rhi/rhi_ray_tracing_program.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::IBackendDevice,
                        grassland::rhi::IObject,
                        "6d6cc71b-9a19-4b0d-ac68-84d86e402f5d")

namespace grassland::rhi::backend {

class IBackendDevice : public IObject {
 public:
  virtual const PhysicalDevice &GetPhysicalDevice() const noexcept = 0;

 protected:
  virtual bool WaitNativeEvents(const std::vector<Event> &events, bool wait_all) const = 0;
  virtual void WaitForProgress(const std::vector<Event> &next) = 0;
  virtual void InterruptCollector() = 0;
};

}  // namespace grassland::rhi::backend
