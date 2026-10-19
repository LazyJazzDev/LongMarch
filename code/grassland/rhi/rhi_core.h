#pragma once
#include "grassland/rhi/rhi_object.h"

namespace grassland::rhi {
class IDevice;

struct DeviceFeatureRequirement {
  // Filters only; every supported RHI feature is enabled during creation.
  bool ray_tracing_pipeline{};
  bool ray_query{};
  bool geometry_shader{};
};

class ICore : public IObject {
 public:
  virtual std::vector<PhysicalDevice> GetPhysicalDevices() const = 0;
  virtual IDevice *CreateDevice(int32_t device_idx) = 0;
  virtual IDevice *CreateDeviceByAutoSelection(DeviceFeatureRequirement requirement = {}) = 0;

 protected:
  ~ICore() override;
};

// Success transfers one owned reference; failure returns nullptr.
ICore *CreateCore(BackendAPI api = BACKEND_API_DEFAULT);

}  // namespace grassland::rhi
