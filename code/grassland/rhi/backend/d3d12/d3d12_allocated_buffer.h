#pragma once
#include "grassland/rhi/backend/d3d12/d3d12_buffer.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::d3d12::ID3D12AllocatedBuffer,
                        grassland::rhi::backend::d3d12::ID3D12Buffer,
                        "f17438f1-4828-4dc2-943d-11e8afe77444")

namespace D3D12MA {
class Allocation;
class Allocator;
}  // namespace D3D12MA

namespace grassland::rhi::backend::d3d12 {
class ID3D12AllocatedBuffer : public ID3D12Buffer {
 public:
  static ID3D12AllocatedBuffer *Create(ID3D12Device *device,
                                       const BufferDesc &desc,
                                       bool upload = false,
                                       bool download = false);

  virtual bool Read(void *data, uint64_t size, uint64_t offset = 0) = 0;
  virtual bool Write(const void *data, uint64_t size, uint64_t offset = 0) = 0;
};

class D3D12AllocatedBufferImpl final : public Object<rhi::IBuffer, ID3D12AllocatedBuffer> {
 public:
  ~D3D12AllocatedBufferImpl() override;
  static ID3D12AllocatedBuffer *Create(ID3D12Device *device,
                                       const BufferDesc &desc,
                                       bool upload = false,
                                       bool download = false);
  ID3D12Resource *NativeBuffer() const override;
  D3D12_RESOURCE_STATES DefaultState() const override;
  bool Read(void *data, uint64_t size, uint64_t offset = 0);
  bool Write(const void *data, uint64_t size, uint64_t offset = 0);

 private:
  D3D12AllocatedBufferImpl(ID3D12Device *device, const BufferDesc &desc);

  ComPtr<ID3D12Resource> buffer_;
  ComPtr<D3D12MA::Allocation> allocation_;
  void *mapped_{};
  bool download_{};

  ID3D12Device *device_;
  BufferDesc desc_;

 public:
  BufferDesc GetDesc() const override {
    return desc_;
  }

  ID3D12Device *Owner() const override;
};

}  // namespace grassland::rhi::backend::d3d12
