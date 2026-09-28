#include "grassland/graphics/backend/d3d12/helper/descriptor_heap.h"

namespace grassland::graphics::backend::d3d12 {

DescriptorHeap::DescriptorHeap(ComPtr<ID3D12DescriptorHeap> descriptor_heap) : descriptor_heap_(descriptor_heap) {
  if (descriptor_heap_) {
    ComPtr<ID3D12Device> device;
    descriptor_heap_->GetDevice(IID_PPV_ARGS(&device));
    descriptor_size_ = device->GetDescriptorHandleIncrementSize(Desc().Type);
  }
}

}  // namespace grassland::graphics::backend::d3d12
