#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/backend/d3d12/d3d12_image.h"

namespace grassland::rhi::backend::d3d12 {
struct D3D12ImageState final {
  D3D12ImageState();
  ~D3D12ImageState();
  ObjectPtr<D3D12DescriptorPool::Lease> attachment_{};
  const void *device_{};
  ID3D12Resource *image_{};
  D3D12_RESOURCE_STATES default_state_{};
};
}  // namespace grassland::rhi::backend::d3d12
