#pragma once
#include "grassland/rhi/backend/backend_device.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

namespace grassland::rhi::backend::d3d12 {

class D3D12Device final : public IDevice {
 public:
  D3D12Device(PhysicalDevice info, const ObjectPtr<ICore> &core, ComPtr<ID3D12Device> device);

 private:
  ComPtr<ID3D12Device> device_;
};

}  // namespace grassland::rhi::backend::d3d12
