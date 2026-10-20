#pragma once
#include "grassland/rhi/backend/d3d12/d3d12_descriptor_pool.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12RTVPool : public D3D12DescriptorPool {
 public:
  explicit D3D12RTVPool(ID3D12Device *device) : D3D12DescriptorPool(device, D3D12_DESCRIPTOR_HEAP_TYPE_RTV) {
  }
};
}  // namespace grassland::rhi::backend::d3d12
