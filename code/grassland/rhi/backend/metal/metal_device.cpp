#include "grassland/rhi/backend/metal/metal_device.h"

#include "grassland/rhi/backend/backend_core.h"

namespace grassland::rhi::backend::metal {

MetalDevice::MetalDevice(PhysicalDevice info, ICore *core, NS::SharedPtr<MTL::Device> device)
    : IDevice(std::move(info), core),
      device_(std::move(device)) {
}

MetalDevice::~MetalDevice() = default;

}  // namespace grassland::rhi::backend::metal
