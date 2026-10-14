#pragma once
#include "grassland/rhi/rhi_device.h"
#include "grassland/rhi/rhi_util.h"

namespace grassland::rhi {

struct DeviceFeatureRequirement {
  // Selection filters only; CreateDevice enables all supported RHI features.
  bool ray_tracing_pipeline{false};
  bool ray_query{false};
  bool geometry_shader{false};
};

class Core {
 public:
  // Vector indices are stable device IDs for the lifetime of this Core.
  [[nodiscard]] std::vector<PhysicalDevice> GetPhysicalDevices() const;
  Device CreateDevice(int32_t device_idx) const;
  Device CreateDeviceByAutoSelection(DeviceFeatureRequirement feature_requirement = {}) const;
  explicit operator bool() const noexcept;

 private:
  using Interface = backend::ICore;
  ObjectRef object_;
  // Internal bridge keeps implementation pointers out of the public API.
  friend struct backend::Access;
};

Core CreateCore(BackendAPI api = BACKEND_API_DEFAULT);

}  // namespace grassland::rhi
