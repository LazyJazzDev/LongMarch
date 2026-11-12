#include "grassland/rhi/backend/d3d12/d3d12_allocated_buffer.h"

#include <cstring>

#include "grassland/rhi/backend/d3d12/d3d12_device.h"
// Match the Windows SDK declarations used by the backend.
#define D3D12MA_D3D12_HEADERS_ALREADY_INCLUDED
#include <D3D12MemAlloc.h>
#undef D3D12MA_D3D12_HEADERS_ALREADY_INCLUDED

namespace grassland::rhi::backend::d3d12 {
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

ID3D12AllocatedBuffer *ID3D12AllocatedBuffer::Create(ID3D12Device *device,
                                                     const BufferDesc &desc,
                                                     bool upload,
                                                     bool download) {
  return D3D12AllocatedBufferImpl::Create(device, desc, upload, download);
}

ID3D12Device *D3D12AllocatedBufferImpl::Owner() const {
  return device_;
}

D3D12AllocatedBufferImpl::D3D12AllocatedBufferImpl(ID3D12Device *device, const BufferDesc &desc)
    : device_(device),
      desc_(desc) {
}

D3D12AllocatedBufferImpl::~D3D12AllocatedBufferImpl() {
  if (mapped_)
    buffer_->Unmap(0, nullptr);
  buffer_.Reset();
}

ID3D12Resource *D3D12AllocatedBufferImpl::NativeBuffer() const {
  return buffer_.Get();
}

D3D12_RESOURCE_STATES D3D12AllocatedBufferImpl::DefaultState() const {
  return download_ ? D3D12_RESOURCE_STATE_COPY_DEST
         : mapped_ ? D3D12_RESOURCE_STATE_GENERIC_READ
                   : D3D12_RESOURCE_STATE_COMMON;
}

ID3D12AllocatedBuffer *D3D12AllocatedBufferImpl::Create(ID3D12Device *device,
                                                        const BufferDesc &desc,
                                                        bool upload,
                                                        bool download) {
  if (!desc.size)
    return nullptr;
  auto object = std::unique_ptr<D3D12AllocatedBufferImpl>(new D3D12AllocatedBufferImpl(device, desc));
  D3D12_RESOURCE_DESC info{};
  info.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  info.Width = object->desc_.size;
  info.Flags = (upload || download) ? D3D12_RESOURCE_FLAG_NONE : D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
  info.Height = 1;
  info.DepthOrArraySize = 1;
  info.MipLevels = 1;
  info.SampleDesc.Count = 1;
  info.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  D3D12MA::ALLOCATION_DESC memory{};
  object->download_ = download;
  memory.HeapType = download ? D3D12_HEAP_TYPE_READBACK : upload ? D3D12_HEAP_TYPE_UPLOAD : D3D12_HEAP_TYPE_DEFAULT;
  if (!Check(
          object->device_->Allocator()->CreateResource(&memory, &info,
                                                       download ? D3D12_RESOURCE_STATE_COPY_DEST
                                                       : upload ? D3D12_RESOURCE_STATE_GENERIC_READ
                                                                : D3D12_RESOURCE_STATE_COMMON,
                                                       nullptr, &object->allocation_, IID_PPV_ARGS(&object->buffer_)),
          "Allocate buffer"))
    return nullptr;
  if (upload || download) {
    D3D12_RANGE read{0, download ? static_cast<SIZE_T>(desc.size) : 0};
    if (!Check(object->buffer_->Map(0, &read, &object->mapped_), "Map upload buffer"))
      return nullptr;
  }
  return object.release();
}

bool D3D12AllocatedBufferImpl::Write(const void *data, uint64_t size, uint64_t offset) {
  if (!mapped_ || offset > desc_.size || size > desc_.size - offset)
    return false;
  std::memcpy(static_cast<uint8_t *>(mapped_) + offset, data, static_cast<size_t>(size));
  return true;
}

bool D3D12AllocatedBufferImpl::Read(void *data, uint64_t size, uint64_t offset) {
  if (!download_ || !mapped_ || !data || offset > desc_.size || size > desc_.size - offset)
    return false;
  std::memcpy(data, static_cast<const uint8_t *>(mapped_) + offset, static_cast<size_t>(size));
  return true;
}
}  // namespace grassland::rhi::backend::d3d12
