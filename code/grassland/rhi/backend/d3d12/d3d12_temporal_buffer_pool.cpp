#include "grassland/rhi/backend/d3d12/d3d12_temporal_buffer_pool.h"

#include "grassland/rhi/backend/d3d12/d3d12_device.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12TemporalBufferLeaseImpl final : public Object<ID3D12TemporalBufferLease> {
 public:
  LONGMARCH_RHI_QUERY_INTERFACES(Object<ID3D12TemporalBufferLease>, IBuffer, IBackendBuffer, ID3D12Buffer)

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
  Device *Owner() const override {
    return buffer_->Owner();
  }
};

ID3D12TemporalBufferLease *ID3D12TemporalBufferLease::Create(D3D12TemporalBufferPool *pool,
                                                             ID3D12AllocatedBuffer *buffer,
                                                             size_t level) {
  return new D3D12TemporalBufferLeaseImpl(pool, buffer, level);
}

void ID3D12TemporalBufferLease::ReturnBuffer(D3D12TemporalBufferPool *pool,
                                             ID3D12AllocatedBuffer *buffer,
                                             size_t level) {
  pool->ReleaseBuffer(buffer, level);
}

D3D12TemporalBufferLeaseImpl::D3D12TemporalBufferLeaseImpl(D3D12TemporalBufferPool *pool,
                                                           ID3D12AllocatedBuffer *buffer,
                                                           size_t level)
    : pool_(pool),
      buffer_(buffer),
      level_(level) {
}

D3D12TemporalBufferLeaseImpl::~D3D12TemporalBufferLeaseImpl() {
  ReturnBuffer(pool_, buffer_, level_);
}

BufferDesc D3D12TemporalBufferLeaseImpl::GetDesc() const {
  return buffer_->GetDesc();
}

ID3D12Resource *D3D12TemporalBufferLeaseImpl::NativeBuffer() const {
  return buffer_->NativeBuffer();
}

D3D12_RESOURCE_STATES D3D12TemporalBufferLeaseImpl::DefaultState() const {
  return buffer_->DefaultState();
}

bool D3D12TemporalBufferLeaseImpl::Write(const void *data, uint64_t size, uint64_t offset) {
  return buffer_->Write(data, size, offset);
}

bool D3D12TemporalBufferLeaseImpl::Read(void *data, uint64_t size, uint64_t offset) {
  return buffer_->Read(data, size, offset);
}
}  // namespace grassland::rhi::backend::d3d12

LONGMARCH_RHI_UUID(grassland::rhi::backend::d3d12::ID3D12TemporalBufferLease, "47d93f8f-2736-4b79-87e7-7cadabcf4f7a")
