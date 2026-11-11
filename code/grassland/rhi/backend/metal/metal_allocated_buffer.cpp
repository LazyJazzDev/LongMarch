#include "grassland/rhi/backend/metal/metal_allocated_buffer.h"

#include <cstring>

#include "grassland/rhi/backend/metal/metal_device.h"

namespace grassland::rhi::backend::metal {
class MetalAllocatedBufferImpl final : public Object<IMetalAllocatedBuffer> {
 public:
  LONGMARCH_RHI_QUERY_INTERFACES(Object<IMetalAllocatedBuffer>, IBuffer, IBackendBuffer, IMetalBuffer)

  static IMetalAllocatedBuffer *Create(IMetalDevice *device,
                                       const BufferDesc &desc,
                                       bool upload = false,
                                       bool download = false);
  MTL::IBackendBuffer *NativeBuffer() const override;
  bool Read(void *data, uint64_t size, uint64_t offset = 0);
  bool Write(const void *data, uint64_t size);

 private:
  MetalAllocatedBufferImpl(IMetalDevice *device, const BufferDesc &desc);
  NS::SharedPtr<MTL::IBackendBuffer> buffer_;

 private:
  IMetalDevice *device_;
  BufferDesc desc_;

 public:
  BufferDesc GetDesc() const override {
    return desc_;
  }

  Device *Owner() const override;
};

IMetalAllocatedBuffer *IMetalAllocatedBuffer::Create(IMetalDevice *device,
                                                     const BufferDesc &desc,
                                                     bool upload,
                                                     bool download) {
  return MetalAllocatedBufferImpl::Create(device, desc, upload, download);
}

Device *MetalAllocatedBufferImpl::Owner() const {
  return device_;
}

MetalAllocatedBufferImpl::MetalAllocatedBufferImpl(IMetalDevice *device, const BufferDesc &desc)
    : device_(device),
      desc_(desc) {
}

IMetalAllocatedBuffer *MetalAllocatedBufferImpl::Create(IMetalDevice *device,
                                                        const BufferDesc &desc,
                                                        bool upload,
                                                        bool download) {
  if (!desc.size || desc.size > device->NativeDevice()->maxBufferLength())
    return nullptr;

  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto object = std::unique_ptr<MetalAllocatedBufferImpl>(new MetalAllocatedBufferImpl(device, desc));
  object->buffer_ = NS::TransferPtr(device->NativeDevice()->newBuffer(
      desc.size, (upload || download) ? MTL::ResourceStorageModeShared : MTL::ResourceStorageModePrivate));
  if (!object->buffer_) {
    ReportError(-1, "Metal buffer allocation failed.");
    return nullptr;
  }
  return object.release();
}

MTL::IBackendBuffer *MetalAllocatedBufferImpl::NativeBuffer() const {
  return buffer_.get();
}

bool MetalAllocatedBufferImpl::Write(const void *data, uint64_t size) {
  if (!data || size > desc_.size || buffer_->storageMode() != MTL::StorageModeShared)
    return false;
  std::memcpy(buffer_->contents(), data, static_cast<size_t>(size));
  return true;
}

bool MetalAllocatedBufferImpl::Read(void *data, uint64_t size, uint64_t offset) {
  if (!data || offset > desc_.size || size > desc_.size - offset || buffer_->storageMode() != MTL::StorageModeShared)
    return false;
  std::memcpy(data, static_cast<const uint8_t *>(buffer_->contents()) + offset, static_cast<size_t>(size));
  return true;
}
}  // namespace grassland::rhi::backend::metal

LONGMARCH_RHI_UUID(grassland::rhi::backend::metal::IMetalAllocatedBuffer, "19d20694-8b38-47a4-a88f-56d5828c0e9d")
