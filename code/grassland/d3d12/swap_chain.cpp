#include "grassland/d3d12/swap_chain.h"

namespace CD::d3d12 {

SwapChain::SwapChain(const ComPtr<IDXGISwapChain3> &swap_chain) : swap_chain_(swap_chain) {
  ComPtr<ID3D12Device> device;
  swap_chain_->GetDevice(IID_PPV_ARGS(&device));
  DXGI_SWAP_CHAIN_DESC desc;
  swap_chain_->GetDesc(&desc);
  back_buffers_.resize(desc.BufferCount);
  width = desc.BufferDesc.Width;
  height = desc.BufferDesc.Height;

  back_buffer_format_ = desc.BufferDesc.Format;

  D3D12_DESCRIPTOR_HEAP_DESC rtv_heap_desc = {};
  rtv_heap_desc.NumDescriptors = desc.BufferCount;
  rtv_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
  rtv_heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;

  device->CreateDescriptorHeap(&rtv_heap_desc, IID_PPV_ARGS(&rtv_heap_));

  rtv_descriptor_size_ = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

  for (uint32_t i = 0; i < desc.BufferCount; i++) {
    swap_chain_->GetBuffer(i, IID_PPV_ARGS(&back_buffers_[i]));
    CD3DX12_CPU_DESCRIPTOR_HANDLE handle(rtv_heap_->GetCPUDescriptorHandleForHeapStart(), i, rtv_descriptor_size_);
    device->CreateRenderTargetView(back_buffers_[i].Get(), nullptr, handle);
  }
}

DXGI_FORMAT SwapChain::BackBufferFormat() const {
  return back_buffer_format_;
}

}  // namespace CD::d3d12
