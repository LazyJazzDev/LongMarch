#pragma once
#include "sparkium/backend/common/compute_device.h"

namespace sparkium::backend {
class CpuDevice final : public ComputeDevice {
 public:
  explicit CpuDevice(const Settings &settings) : ComputeDevice(settings) {
  }

  RenderBackend API() const override {
    return RenderBackend::CPU;
  }

  int GetPhysicalDeviceProperties(PhysicalDeviceProperties *properties = nullptr) override;
  int InitializeLogicalDevice(int index) override;

  void WaitGPU() override {
  }
};
}  // namespace sparkium::backend
