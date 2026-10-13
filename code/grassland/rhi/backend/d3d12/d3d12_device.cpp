#include "grassland/rhi/backend/d3d12/d3d12_device.h"

namespace grassland::rhi::backend::d3d12 {

D3D12Device::D3D12Device(PhysicalDevice info, const ObjectPtr<ICore> &core, ComPtr<ID3D12Device> device)
    : IDevice(std::move(info), core),
      device_(std::move(device)) {
}

}  // namespace grassland::rhi::backend::d3d12
