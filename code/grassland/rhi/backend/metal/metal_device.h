#pragma once

#include <Metal/Metal.hpp>

#include "grassland/rhi/backend/backend_device.h"

namespace grassland::rhi::backend::metal {

class MetalDevice final : public Device {
 public:
  MetalDevice(PhysicalDevice info, Core *core, NS::SharedPtr<MTL::Device> device);
  ~MetalDevice() override;

  MTL::Device *NativeDevice() const noexcept;

 private:
  NS::SharedPtr<MTL::Device> device_;
};

}  // namespace grassland::rhi::backend::metal
