#include "grassland/rhi/backend/d3d12/d3d12_rtv_pool.h"

namespace grassland::rhi::backend::d3d12 {
D3D12RTVPool::D3D12RTVPool(ID3D12Device *device)
    : device_(device),
      stride_(device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV)) {
}

bool D3D12RTVPool::Acquire(ID3D12Resource *image, uint64_t &slot, D3D12_CPU_DESCRIPTOR_HANDLE &handle) {
  std::lock_guard<std::mutex> lock(mutex_);
  size_t page = 0;
  for (; page < pages_.size(); ++page)
    if (pages_[page].occupied != UINT64_MAX)
      break;
  if (page == pages_.size()) {
    D3D12_DESCRIPTOR_HEAP_DESC desc{};
    desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    desc.NumDescriptors = 64;
    Page fresh;
    if (!Check(device_->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&fresh.heap)), "Create RTV pool page"))
      return false;
    pages_.push_back(std::move(fresh));
  }
  auto &entry = pages_[page];
  uint32_t bit = 0;
  while (entry.occupied & (uint64_t{1} << bit))
    ++bit;
  entry.occupied |= uint64_t{1} << bit;
  slot = page * 64 + bit;
  handle = entry.heap->GetCPUDescriptorHandleForHeapStart();
  handle.ptr += bit * stride_;
  device_->CreateRenderTargetView(image, nullptr, handle);
  return true;
}

void D3D12RTVPool::Release(uint64_t slot) {
  std::lock_guard<std::mutex> lock(mutex_);
  pages_[slot / 64].occupied &= ~(uint64_t{1} << (slot % 64));
}
}  // namespace grassland::rhi::backend::d3d12
