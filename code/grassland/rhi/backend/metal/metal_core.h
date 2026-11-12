#pragma once
#include <Metal/Metal.hpp>

#include "grassland/rhi/backend/backend_core.h"
#include "grassland/rhi/backend/metal/metal_util.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::metal::IMetalCore,
                        grassland::rhi::IObject,
                        "30ef4d7e-ae80-4882-b15d-023c44be58cf")

namespace grassland::rhi::backend::metal {

class IMetalCore : public IObject {
 public:
  static rhi::ICore *CreateMetalCore(const CoreDesc &desc);
  virtual bool DebugEnabled() const = 0;
};

}  // namespace grassland::rhi::backend::metal

namespace grassland::rhi::backend::metal {
class MetalCoreImpl final : public Object<rhi::ICore, IMetalCore> {
 public:
  static rhi::ICore *CreateMetalCore(const CoreDesc &desc);
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
  ~MetalCoreImpl() override = default;

  rhi::IDevice *CreateDeviceByAutoSelection(DeviceFeatureRequirement requirement = {}) override {
    return backend::CreateDeviceByAutoSelection(this, requirement);
  }
};
}  // namespace grassland::rhi::backend::metal
