#pragma once
#include "grassland/rhi/backend/d3d12/d3d12_buffer.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12UploadPool;
class D3D12AllocatedBuffer;

// Borrows pool-owned storage. The pool and its Device must outlive every lease.
class D3D12UploadBufferLease final : public D3D12Buffer {
 public:
  D3D12UploadBufferLease(D3D12UploadPool *pool, D3D12AllocatedBuffer *buffer);
  ~D3D12UploadBufferLease() override;
  BufferDesc GetDesc() const override;
  ID3D12Resource *NativeBuffer() const override;
  D3D12_RESOURCE_STATES DefaultState() const override;
  bool Write(const void *data, uint64_t size);

 private:
  D3D12UploadPool *pool_;
  D3D12AllocatedBuffer *buffer_;
};
}  // namespace grassland::rhi::backend::d3d12
