#include "grassland/rhi/backend/metal/metal_device.h"

#include "grassland/rhi/backend/backend_core.h"

namespace grassland::rhi::backend::metal {

MetalDevice::MetalDevice(PhysicalDevice info, Core *core, NS::SharedPtr<MTL::Device> device)
    : Device(std::move(info), core),
      device_(std::move(device)) {
}

MetalDevice::~MetalDevice() = default;

MTL::Device *MetalDevice::NativeDevice() const noexcept {
  return device_.get();
}

}  // namespace grassland::rhi::backend::metal
