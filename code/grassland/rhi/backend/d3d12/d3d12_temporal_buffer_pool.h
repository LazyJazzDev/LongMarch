#pragma once
#include "grassland/rhi/backend/backend_temporal_buffer_pool.h"
#include "grassland/rhi/backend/d3d12/d3d12_allocated_buffer.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::d3d12::ID3D12TemporalBufferLease,
                        grassland::rhi::backend::d3d12::ID3D12Buffer,
                        "47d93f8f-2736-4b79-87e7-7cadabcf4f7a")

namespace grassland::rhi::backend::d3d12 {
using D3D12TemporalBufferPool = TemporalBufferPool<ID3D12Device, ID3D12AllocatedBuffer, ID3D12TemporalBufferLease>;

class ID3D12TemporalBufferLease : public ID3D12Buffer {
 public:
  static ID3D12TemporalBufferLease *Create(D3D12TemporalBufferPool *pool, ID3D12AllocatedBuffer *buffer, size_t level);

  virtual bool Read(void *data, uint64_t size, uint64_t offset = 0) = 0;
  virtual bool Write(const void *data, uint64_t size, uint64_t offset = 0) = 0;

 protected:
  static void ReturnBuffer(D3D12TemporalBufferPool *pool, ID3D12AllocatedBuffer *buffer, size_t level);
};

class D3D12TemporalBufferLeaseImpl final : public Object<rhi::IBuffer, ID3D12TemporalBufferLease> {
 public:
  D3D12TemporalBufferLeaseImpl(D3D12TemporalBufferPool *pool, ID3D12AllocatedBuffer *buffer, size_t level);
  ~D3D12TemporalBufferLeaseImpl() override;
  BufferDesc GetDesc() const override;
  ID3D12Resource *NativeBuffer() const override;
  D3D12_RESOURCE_STATES DefaultState() const override;
  bool Read(void *data, uint64_t size, uint64_t offset = 0);
  bool Write(const void *data, uint64_t size, uint64_t offset = 0);

 private:
  D3D12TemporalBufferPool *pool_;
  ID3D12AllocatedBuffer *buffer_;
  size_t level_;

 public:
  ID3D12Device *Owner() const override {
    return buffer_->Owner();
  }
};

}  // namespace grassland::rhi::backend::d3d12
