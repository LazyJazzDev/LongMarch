#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend::metal {
class MetalCore;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::metal::MetalCore)

#include <Metal/Metal.hpp>

#include "grassland/rhi/backend/backend_core.h"

namespace grassland::rhi::backend::metal {

class MetalCore final : public Core {
 public:
  LONGMARCH_RHI_QUERY(MetalCore, Core)

  static MetalCore *CreateMetalCore(const CoreDesc &desc);
  std::vector<PhysicalDevice> GetPhysicalDevices() const override;
  rhi::IDevice *CreateDevice(int32_t device_idx) override;

  bool DebugEnabled() const;

 private:
  explicit MetalCore(const CoreDesc &desc);
  const bool enable_debug_;

  struct Adapter {
    NS::SharedPtr<MTL::Device> native;
    PhysicalDevice info;
  };

  std::vector<Adapter> adapters_;
};

}  // namespace grassland::rhi::backend::metal
