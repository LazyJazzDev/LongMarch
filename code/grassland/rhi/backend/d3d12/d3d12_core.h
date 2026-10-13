#pragma once
#include "grassland/rhi/backend/backend_core.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

namespace grassland::rhi::backend::d3d12 {

class D3D12Core final : public ICore {
 public:
  static std::unique_ptr<D3D12Core> CreateD3D12Core();
  std::vector<PhysicalDevice> GetPhysicalDevices() const override;
  std::unique_ptr<IDevice> CreateDevice(int32_t device_idx, const ObjectRef &owner) const override;

 private:
  struct Adapter {
    ComPtr<IDXGIAdapter1> native;
    PhysicalDevice info;
  };

  ComPtr<IDXGIFactory4> factory_;
  std::vector<Adapter> adapters_;
};

}  // namespace grassland::rhi::backend::d3d12
