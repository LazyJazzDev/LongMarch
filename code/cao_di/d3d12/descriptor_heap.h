#pragma once
#include "cao_di/d3d12/device.h"

namespace CD::d3d12 {

class DescriptorHeap {
 public:
  DescriptorHeap(ComPtr<ID3D12DescriptorHeap> descriptor_heap);

  D3D12_DESCRIPTOR_HEAP_DESC Desc() const {
    return descriptor_heap_->GetDesc();
  }

  uint32_t NumDescriptors() const {
    return Desc().NumDescriptors;
  }

  D3D12_DESCRIPTOR_HEAP_TYPE Type() const {
    return Desc().Type;
  }

  uint32_t DescriptorSize() const {
    return descriptor_size_;
  }

  ID3D12DescriptorHeap *Handle() const {
    return descriptor_heap_.Get();
  }

  CD3DX12_CPU_DESCRIPTOR_HANDLE CPUHandle(uint32_t index) const {
    return CD3DX12_CPU_DESCRIPTOR_HANDLE(descriptor_heap_->GetCPUDescriptorHandleForHeapStart(), index,
                                         descriptor_size_);
  }

  CD3DX12_GPU_DESCRIPTOR_HANDLE GPUHandle(uint32_t index) const {
    return CD3DX12_GPU_DESCRIPTOR_HANDLE(descriptor_heap_->GetGPUDescriptorHandleForHeapStart(), index,
                                         descriptor_size_);
  }

 private:
  uint32_t descriptor_size_{};
  ComPtr<ID3D12DescriptorHeap> descriptor_heap_;
};

}  // namespace CD::d3d12
