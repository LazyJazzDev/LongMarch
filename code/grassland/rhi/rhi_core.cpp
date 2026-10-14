#include "grassland/rhi/rhi_core.h"

#include <new>

#include "grassland/rhi/backend/backend.h"

namespace grassland::rhi {

Core::Core() noexcept = default;

Core CreateCore(BackendAPI api) {
  try {
    return backend::Access::Adopt<Core>(backend::CreateCore(api));
  } catch (const std::bad_alloc &) {
    // Standard-library allocation is the exceptional boundary; native failures return empty directly.
    return {};
  }
}

std::vector<PhysicalDevice> Core::GetPhysicalDevices() const {
  if (!*this) {
    ReportError(-1, "GetPhysicalDevices called on an empty Core.");
    return {};
  }
  return backend::Access::As<backend::ICore>(*this).GetPhysicalDevices();
}

Device Core::CreateDevice(int32_t device_idx) const {
  if (!*this) {
    ReportError(-1, "CreateDevice called on an empty Core.");
    return {};
  }
  try {
    const auto &core = backend::Access::As<backend::ICore>(*this);
    const auto devices = core.GetPhysicalDevices();
    if (device_idx < 0 || static_cast<size_t>(device_idx) >= devices.size()) {
      ReportError(-1, "Physical device ID is out of range.");
      return {};
    }
    return backend::Access::Adopt<Device>(core.CreateDevice(device_idx, &backend::Access::As<backend::ICore>(*this)));
  } catch (const std::bad_alloc &) {
    // Standard-library allocation is the exceptional boundary; native failures return empty directly.
    return {};
  }
}

Device Core::CreateDeviceByAutoSelection(DeviceFeatureRequirement requirement) const {
  const int32_t device_idx = backend::SelectPhysicalDevice(GetPhysicalDevices(), requirement);
  if (device_idx < 0) {
    ReportError(-1, "No physical device satisfies the requested RHI features.");
    return {};
  }
  return CreateDevice(device_idx);
}

}  // namespace grassland::rhi
