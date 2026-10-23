#pragma once
#include "grassland/rhi/backend/backend_core.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

namespace grassland::rhi::backend::d3d12 {

class D3D12Core final : public Core {
 public:
  static D3D12Core *CreateD3D12Core(const CoreDesc &desc);
  ~D3D12Core() override;
  std::vector<PhysicalDevice> GetPhysicalDevices() const override;
  rhi::IDevice *CreateDevice(int32_t device_idx) override;

 private:
  D3D12Core();

  struct Adapter {
    ComPtr<IDXGIAdapter1> native;
    PhysicalDevice info;
  };

  bool registered_{};
  ComPtr<IDXGIFactory4> factory_;
  std::vector<Adapter> adapters_;
};

}  // namespace grassland::rhi::backend::d3d12
