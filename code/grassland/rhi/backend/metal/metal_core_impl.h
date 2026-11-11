#pragma once
#include "grassland/rhi/backend/metal/metal_core.h"
#include "grassland/rhi/backend/metal/metal_util.h"

namespace grassland::rhi::backend::metal {
class MetalCoreImpl final : public Object<IMetalCore> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(rhi::ICore, IMetalCore, IBackendCore, rhi::ICore)
  static IMetalCore *CreateMetalCore(const CoreDesc &desc);
  std::vector<PhysicalDevice> GetPhysicalDevices() const override;
  rhi::IDevice *CreateDevice(int32_t device_idx) override;
  bool DebugEnabled() const;

 private:
  explicit MetalCoreImpl(const CoreDesc &desc);
  const bool enable_debug_;

  struct Adapter {
    NS::SharedPtr<MTL::Device> native;
    PhysicalDevice info;
  };

  std::vector<Adapter> adapters_;

 public:
  ~MetalCoreImpl() override {
  }

 public:
  rhi::IDevice *CreateDeviceByAutoSelection(DeviceFeatureRequirement requirement = {}) override {
    return IBackendCore::CommonCreateDeviceByAutoSelection(requirement);
  }
};
}  // namespace grassland::rhi::backend::metal
