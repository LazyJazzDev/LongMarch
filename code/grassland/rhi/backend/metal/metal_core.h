#pragma once

#include <Metal/Metal.hpp>

#include "grassland/rhi/backend/backend_core.h"

namespace grassland::rhi::backend::metal {

class MetalCore final : public Core {
 public:
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
