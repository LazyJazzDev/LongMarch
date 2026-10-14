#pragma once

#include <Metal/Metal.hpp>

#include "grassland/rhi/backend/backend_device.h"

namespace grassland::rhi::backend::metal {

class MetalDevice final : public IDevice {
 public:
  MetalDevice(PhysicalDevice info, ICore *core, NS::SharedPtr<MTL::Device> device);
  ~MetalDevice() override;

  MTL::Device *NativeDevice() const noexcept {
    return device_.get();
  }

 private:
  NS::SharedPtr<MTL::Device> device_;
};

}  // namespace grassland::rhi::backend::metal
