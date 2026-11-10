#pragma once
#include "grassland/rhi/backend/backend_cuda_buffer.h"
#if defined(LONGMARCH_CUDA_RUNTIME)
#include "grassland/rhi/backend/d3d12/d3d12_buffer.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12Device;

// Implements the most-derived interface directly: one interface chain and reference count.
class D3D12CUDABuffer final : public Object<ICUDABuffer>, public D3D12Buffer {
 public:
  static D3D12CUDABuffer *Create(D3D12Device *device, const BufferDesc &desc);
  BufferDesc GetDesc() const override;
  void *GetDevicePointer() const override;
  ID3D12Resource *NativeBuffer() const override;
  D3D12_RESOURCE_STATES DefaultState() const override;

 private:
  D3D12CUDABuffer(D3D12Device *device, const BufferDesc &desc, int cuda_device);
  ComPtr<ID3D12Resource> resource_;
  CUDAMapping mapping_;
};
}  // namespace grassland::rhi::backend::d3d12
#endif
