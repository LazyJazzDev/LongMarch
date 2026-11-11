#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/backend/d3d12/d3d12_core.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12CoreImpl final : public Object<ID3D12Core> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(rhi::ICore, ID3D12Core, IBackendCore, rhi::ICore)
  static ID3D12Core *CreateD3D12Core(const CoreDesc &desc);
  ~D3D12CoreImpl() override;
  std::vector<PhysicalDevice> GetPhysicalDevices() const override;
  rhi::IDevice *CreateDevice(int32_t device_idx) override;

 private:
  D3D12CoreImpl();

  struct Adapter {
    ComPtr<IDXGIAdapter1> native;
    PhysicalDevice info;
  };

  ComPtr<ID3D12Debug4> debug_controller_;
  ComPtr<IDXGIFactory4> factory_;
  std::vector<Adapter> adapters_;

 public:
  rhi::IDevice *CreateDeviceByAutoSelection(DeviceFeatureRequirement requirement = {}) override {
    return IBackendCore::CommonCreateDeviceByAutoSelection(requirement);
  }
};
}  // namespace grassland::rhi::backend::d3d12
