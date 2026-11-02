#pragma once
#include "grassland/rhi/backend/backend_cuda_interop.h"
#if defined(LONGMARCH_CUDA_RUNTIME)
#include "grassland/rhi/backend/d3d12/d3d12_buffer.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12CUDABuffer final : public D3D12Buffer, public ICUDABuffer {
 public:
  static D3D12CUDABuffer *Create(D3D12Device *device, const BufferDesc &desc);
  ID3D12Resource *NativeBuffer() const override;
  D3D12_RESOURCE_STATES DefaultState() const override;
  void *GetDevicePointer() const override;

 private:
  D3D12CUDABuffer(D3D12Device *device, const BufferDesc &desc);
  ComPtr<ID3D12Resource> resource_;
  CUDAMapping mapping_;
};

class D3D12CUDAInteropTimeline final : public CUDAInteropTimeline {
 public:
  static D3D12CUDAInteropTimeline *Create(D3D12Device *device);
  ~D3D12CUDAInteropTimeline() override;
  ID3D12Fence *NativeFence() const;

 private:
  explicit D3D12CUDAInteropTimeline(D3D12Device *device);
  ComPtr<ID3D12Fence> fence_;
};
}  // namespace grassland::rhi::backend::d3d12
#endif
