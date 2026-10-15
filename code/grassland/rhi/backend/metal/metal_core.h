#pragma once

#include <Metal/Metal.hpp>

#include "grassland/rhi/backend/backend_core.h"

namespace grassland::rhi::backend::metal {

class MetalCore final : public Core {
 public:
  static std::unique_ptr<MetalCore> CreateMetalCore();
  std::vector<PhysicalDevice> GetPhysicalDevices() const override;
  rhi::IDevice *CreateDevice(int32_t device_idx) override;

 private:
  struct Adapter {
    NS::SharedPtr<MTL::Device> native;
    PhysicalDevice info;
  };

  std::vector<Adapter> adapters_;
};

}  // namespace grassland::rhi::backend::metal
