#pragma once
#include "grassland/rhi/rhi_object.h"

namespace grassland::rhi {
class IDevice;

struct CoreDesc {
#if !defined(NDEBUG)
  bool enable_debug{true};
#else
  bool enable_debug{false};
#endif
};

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
// D3D12/Vulkan native validation; unavailable requested validation fails creation.
// Concurrent D3D12 Cores must agree on enable_debug (a process-wide setting).
// Metal validation remains controlled by the native environment/tools.
ICore *CreateCore(BackendAPI api = BACKEND_API_DEFAULT, const CoreDesc &desc = {});

}  // namespace grassland::rhi
