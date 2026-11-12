#pragma once
#include "grassland/rhi/backend/backend_cuda_buffer.h"
#include "grassland/rhi/backend/d3d12/d3d12_buffer.h"
#if defined(LONGMARCH_CUDA_RUNTIME)
namespace grassland::rhi::backend::d3d12 {
class D3D12CUDABufferImpl final : public Object<rhi::ICUDABuffer, ID3D12Buffer> {
 public:
  static rhi::IBuffer *Create(ID3D12Device *device, const BufferDesc &desc);
  ~D3D12CUDABufferImpl() override = default;
  BufferDesc GetDesc() const override;
  void *GetDevicePointer() const override;

  ID3D12Device *Owner() const override {
    return owner_;
  }

  ID3D12Resource *NativeBuffer() const override;
  D3D12_RESOURCE_STATES DefaultState() const override;

 private:
  D3D12CUDABufferImpl(ID3D12Device *device, const BufferDesc &desc, int cuda_device);
  ID3D12Device *owner_;
  BufferDesc desc_;
  ComPtr<ID3D12Resource> resource_;
  CUDAMapping mapping_;
};
}  // namespace grassland::rhi::backend::d3d12
#endif
