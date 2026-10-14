#include "grassland/rhi/rhix/rhix_core.h"

namespace grassland::rhix {
Core::Core() noexcept = default;

Core::Core(rhi::ICore *object) noexcept : ObjectContainer(object) {
}

rhi::ICore *Core::Get() const noexcept {
  return static_cast<rhi::ICore *>(GetObject());
}

Core CreateCore(rhi::BackendAPI api) {
  return rhi::CreateCore(api);
}

std::vector<rhi::PhysicalDevice> Core::GetPhysicalDevices() const {
  return *this ? Get()->GetPhysicalDevices() : std::vector<rhi::PhysicalDevice>{};
}

Device Core::CreateDevice(int32_t id) const {
  return *this ? Get()->CreateDevice(id) : nullptr;
}

Device Core::CreateDeviceByAutoSelection(rhi::DeviceFeatureRequirement requirement) const {
  return *this ? Get()->CreateDeviceByAutoSelection(requirement) : nullptr;
}
}  // namespace grassland::rhix
