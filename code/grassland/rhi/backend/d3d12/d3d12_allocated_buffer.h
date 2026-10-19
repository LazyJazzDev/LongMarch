#pragma once
#include "grassland/rhi/backend/d3d12/d3d12_buffer.h"

namespace D3D12MA {
class Allocation;
class Allocator;
}  // namespace D3D12MA

namespace grassland::rhi::backend::d3d12 {
class D3D12AllocatedBuffer final : public D3D12Buffer {
 public:
  D3D12AllocatedBuffer(D3D12Device *device, const BufferDesc &desc);
  ~D3D12AllocatedBuffer() override;
  bool Initialize(bool upload = false);
  ID3D12Resource *NativeBuffer() const override;
  D3D12_RESOURCE_STATES DefaultState() const override;
  bool Write(const void *data, uint64_t size);

 private:
  ComPtr<ID3D12Resource> buffer_;
  ComPtr<D3D12MA::Allocation> allocation_;
  void *mapped_{};
};
}  // namespace grassland::rhi::backend::d3d12
