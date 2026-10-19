#include "grassland/rhi/backend/d3d12/d3d12_upload_buffer_lease.h"

#include "grassland/rhi/backend/d3d12/d3d12_device.h"
#include "grassland/rhi/backend/d3d12/d3d12_upload_pool.h"

namespace grassland::rhi::backend::d3d12 {
D3D12UploadBufferLease::D3D12UploadBufferLease(D3D12UploadPool *pool, D3D12AllocatedBuffer *buffer)
    : D3D12Buffer(static_cast<D3D12Device *>(buffer->Owner()), buffer->GetDesc()),
      pool_(pool),
      buffer_(buffer) {
}

D3D12UploadBufferLease::~D3D12UploadBufferLease() {
  pool_->ReleaseBuffer(buffer_);
}

BufferDesc D3D12UploadBufferLease::GetDesc() const {
  return buffer_->GetDesc();
}

ID3D12Resource *D3D12UploadBufferLease::NativeBuffer() const {
  return buffer_->NativeBuffer();
}

D3D12_RESOURCE_STATES D3D12UploadBufferLease::DefaultState() const {
  return buffer_->DefaultState();
}

bool D3D12UploadBufferLease::Write(const void *data, uint64_t size) {
  return buffer_->Write(data, size);
}
}  // namespace grassland::rhi::backend::d3d12
