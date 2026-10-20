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

D3D12AllocatedBuffer *D3D12AllocatedBuffer::Create(D3D12Device *device, const BufferDesc &desc, bool upload) {
  if (!desc.size)
    return nullptr;
  try {
    auto object = std::unique_ptr<D3D12AllocatedBuffer>(new D3D12AllocatedBuffer(device, desc));
    D3D12_RESOURCE_DESC info{};
    info.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    info.Width = object->desc_.size;
    info.Flags = upload ? D3D12_RESOURCE_FLAG_NONE : D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    info.Height = 1;
    info.DepthOrArraySize = 1;
    info.MipLevels = 1;
    info.SampleDesc.Count = 1;
    info.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    D3D12MA::ALLOCATION_DESC memory{};
    memory.HeapType = upload ? D3D12_HEAP_TYPE_UPLOAD : D3D12_HEAP_TYPE_DEFAULT;
    if (!Check(static_cast<D3D12Device *>(object->device_)
                   ->Allocator()
                   ->CreateResource(&memory, &info,
                                    upload ? D3D12_RESOURCE_STATE_GENERIC_READ : D3D12_RESOURCE_STATE_COMMON, nullptr,
                                    &object->allocation_, IID_PPV_ARGS(&object->buffer_)),
               "Allocate buffer"))
      return nullptr;
    if (upload) {
      D3D12_RANGE read{0, 0};
      if (!Check(object->buffer_->Map(0, &read, &object->mapped_), "Map upload buffer"))
        return nullptr;
    }
    return object.release();
  } catch (const std::bad_alloc &) {
    ReportError(-1, "D3D12AllocatedBuffer creation ran out of memory.");
    return nullptr;
  }
}

bool D3D12AllocatedBuffer::Write(const void *data, uint64_t size) {
  if (!mapped_ || size > desc_.size)
    return false;
  std::memcpy(mapped_, data, static_cast<size_t>(size));
  return true;
}
}  // namespace grassland::rhi::backend::d3d12
