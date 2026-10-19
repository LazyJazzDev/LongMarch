#include "grassland/rhi/backend/d3d12/d3d12_allocated_buffer.h"

#include <cstring>

#include "grassland/rhi/backend/d3d12/d3d12_device.h"
// Match the Windows SDK declarations used by the backend.
#define D3D12MA_D3D12_HEADERS_ALREADY_INCLUDED
#include <D3D12MemAlloc.h>
#undef D3D12MA_D3D12_HEADERS_ALREADY_INCLUDED

namespace grassland::rhi::backend::d3d12 {
D3D12AllocatedBuffer::D3D12AllocatedBuffer(D3D12Device *device, const BufferDesc &desc) : D3D12Buffer(device, desc) {
}

D3D12AllocatedBuffer::~D3D12AllocatedBuffer() {
  if (mapped_)
    buffer_->Unmap(0, nullptr);
  buffer_.Reset();
}

ID3D12Resource *D3D12AllocatedBuffer::NativeBuffer() const {
  return buffer_.Get();
}

D3D12_RESOURCE_STATES D3D12AllocatedBuffer::DefaultState() const {
  return mapped_ ? D3D12_RESOURCE_STATE_GENERIC_READ : D3D12_RESOURCE_STATE_COMMON;
}

bool D3D12AllocatedBuffer::Initialize(bool upload) {
  D3D12_RESOURCE_DESC info{};
  info.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  info.Width = desc_.size;
  info.Height = 1;
  info.DepthOrArraySize = 1;
  info.MipLevels = 1;
  info.SampleDesc.Count = 1;
  info.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  D3D12MA::ALLOCATION_DESC memory{};
  memory.HeapType = upload ? D3D12_HEAP_TYPE_UPLOAD : D3D12_HEAP_TYPE_DEFAULT;
  if (!Check(static_cast<D3D12Device *>(device_)->Allocator()->CreateResource(
                 &memory, &info, upload ? D3D12_RESOURCE_STATE_GENERIC_READ : D3D12_RESOURCE_STATE_COMMON, nullptr,
                 &allocation_, IID_PPV_ARGS(&buffer_)),
             "Allocate buffer"))
    return false;
  if (upload) {
    D3D12_RANGE read{0, 0};
    if (!Check(buffer_->Map(0, &read, &mapped_), "Map upload buffer"))
      return false;
  }
  return true;
}

bool D3D12AllocatedBuffer::Write(const void *data, uint64_t size) {
  if (!mapped_ || size > desc_.size)
    return false;
  std::memcpy(mapped_, data, static_cast<size_t>(size));
  return true;
}
}  // namespace grassland::rhi::backend::d3d12
