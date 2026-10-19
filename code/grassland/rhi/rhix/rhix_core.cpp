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
#if !defined(NDEBUG)
  if (!*this)
    return std::vector<rhi::PhysicalDevice>{};
#endif
  return Get()->GetPhysicalDevices();
}

Device Core::CreateDevice(int32_t id) const {
#if !defined(NDEBUG)
  if (!*this)
    return nullptr;
#endif
  return Get()->CreateDevice(id);
}

Device Core::CreateDeviceByAutoSelection(rhi::DeviceFeatureRequirement requirement) const {
#if !defined(NDEBUG)
  if (!*this)
    return nullptr;
#endif
  return Get()->CreateDeviceByAutoSelection(requirement);
}
}  // namespace grassland::rhix
