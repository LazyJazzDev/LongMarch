#pragma once
#include <Metal/Metal.hpp>

#include "grassland/rhi/backend/backend_core.h"
#include "grassland/rhi/backend/metal/metal_util.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::metal::IMetalCore)

namespace grassland::rhi::backend::metal {

class IMetalCore : public IBackendCore {
 public:
  static IMetalCore *CreateMetalCore(const CoreDesc &desc);
  virtual std::vector<PhysicalDevice> GetPhysicalDevices() const = 0;
  virtual rhi::IDevice *CreateDevice(int32_t device_idx) = 0;
  virtual bool DebugEnabled() const = 0;
};

}  // namespace grassland::rhi::backend::metal
