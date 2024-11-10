#pragma once
#include "grassland/d3d12/adapter.h"

namespace grassland::d3d12 {
class Device {
 public:
  Device(const Adapter &adapter,
         D3D_FEATURE_LEVEL feature_level,
         ComPtr<ID3D12Device> device);

  ID3D12Device *Handle() const {
    return device_.Get();
  }

  const Adapter &Adapter() const {
    return adapter_;
  }

  D3D_FEATURE_LEVEL FeatureLevel() const {
    return feature_level_;
  }

  HRESULT CreateCommandQueue(D3D12_COMMAND_LIST_TYPE type,
                             double_ptr<CommandQueue> pp_command_queue);

  HRESULT CreateCommandAllocator(
      D3D12_COMMAND_LIST_TYPE type,
      double_ptr<CommandAllocator> pp_command_allocator);

  HRESULT CreateDescriptorHeap(const D3D12_DESCRIPTOR_HEAP_DESC &desc,
                               double_ptr<DescriptorHeap> pp_descriptor_heap);

  HRESULT CreateDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE type,
                               uint32_t num_descriptors,
                               double_ptr<DescriptorHeap> pp_descriptor_heap);

  HRESULT CreateFence(double_ptr<Fence> pp_fence);

 private:
  class Adapter adapter_;
  ComPtr<ID3D12Device> device_;
  D3D_FEATURE_LEVEL feature_level_;
};
}  // namespace grassland::d3d12
