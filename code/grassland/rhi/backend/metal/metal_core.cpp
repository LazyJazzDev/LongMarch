#include "grassland/rhi/backend/metal/metal_core.h"

#include <TargetConditionals.h>
#include <sys/sysctl.h>

#include "grassland/rhi/backend/metal/metal_device.h"

namespace grassland::rhi::backend::metal {

std::unique_ptr<MetalCore> MetalCore::CreateMetalCore() {
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto core = std::make_unique<MetalCore>();
  uint64_t host_memory = 0;
  size_t size = sizeof(host_memory);
  if (sysctlbyname("hw.memsize", &host_memory, &size, nullptr, 0) != 0)
    host_memory = 0;  // Unknown capacity does not prevent device creation.

  auto add_device = [&](MTL::Device *device) {
    PhysicalDevice info{};
    const auto name = device->name();
    info.name = name ? name->utf8String() : "Metal device";
    info.type = device->hasUnifiedMemory() ? PHYSICAL_DEVICE_TYPE_INTEGRATED : PHYSICAL_DEVICE_TYPE_DEDICATED;
    info.unified_memory = host_memory;
    // Metal exposes a recommended working-set budget, not portable VRAM capacity.
    // Leave dedicated_memory unknown (zero) instead of reporting that budget as VRAM.
    info.ray_query_feature = device->supportsRaytracing();
    // Metal intersectors run inside compute/render shaders. They do not provide
    // DXR/Vulkan ray-generation/miss/hit pipeline stages or geometry shaders.
    info.ray_tracing_pipeline_feature = false;
    info.geometry_shader_feature = false;
    core->adapters_.push_back({NS::RetainPtr(device), std::move(info)});
  };

#if TARGET_OS_OSX
  auto devices = NS::TransferPtr(MTL::CopyAllDevices());
  if (!devices) {
    ReportError(-1, "MTLCopyAllDevices failed.");
    return {};
  }
  for (NS::UInteger i = 0; i < devices->count(); ++i) {
    auto device = devices->object<MTL::Device>(i);
    if (!device) {
      ReportError(-1, "Metal device enumeration returned a null device.");
      return {};
    }
    add_device(device);
  }
#else
  auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  if (!device) {
    ReportError(-1, "MTLCreateSystemDefaultDevice failed.");
    return {};
  }
  add_device(device.get());
#endif
  return core;
}

std::vector<PhysicalDevice> MetalCore::GetPhysicalDevices() const {
  std::vector<PhysicalDevice> devices;
  for (const auto &adapter : adapters_)
    devices.push_back(adapter.info);
  return devices;
}

std::unique_ptr<Device> MetalCore::CreateNativeDevice(int32_t device_idx, Core *owner) const {
#if !defined(NDEBUG)
  if (device_idx < 0 || static_cast<size_t>(device_idx) >= adapters_.size()) {
    ReportError(-1, "Physical device ID is out of range.");
    return {};
  }
#endif
  const auto &adapter = adapters_[device_idx];
  // Metal's enumerated MTLDevice is already usable; no feature enable mask or
  // separate logical-device creation exists. Each wrapper owns a native retain.
  return std::make_unique<MetalDevice>(adapter.info, owner, adapter.native);
}

}  // namespace grassland::rhi::backend::metal
