#pragma once
#include "grassland/rhi/rhi.h"
#include "grassland/rhi/rhi_util.h"
#include "grassland/rhi/rhix/rhix_device.h"
#include "grassland/rhi/rhix/rhix_object_container.h"

namespace grassland::rhix {

class Core : public ObjectContainer {
 public:
  Core(rhi::ICore *object) noexcept;
  rhi::ICore *Get() const noexcept;
  Core() noexcept;
  rhi::BackendAPI GetBackendAPI() const;
  // Vector indices are stable device IDs for the lifetime of this Core.
  [[nodiscard]] std::vector<rhi::PhysicalDevice> GetPhysicalDevices() const;
  Device CreateDevice(int32_t device_idx) const;
  Device CreateDeviceByAutoSelection(rhi::DeviceFeatureRequirement feature_requirement = {}) const;
};

Core CreateCore(rhi::BackendAPI api = rhi::BACKEND_API_DEFAULT, const rhi::CoreDesc &desc = {});

}  // namespace grassland::rhix
