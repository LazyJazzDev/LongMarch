#include "grassland/rhi/backend/metal/metal_allocated_buffer.h"

#include <cstring>

#include "grassland/rhi/backend/metal/metal_device.h"

namespace grassland::rhi::backend::metal {
MetalAllocatedBuffer::MetalAllocatedBuffer(MetalDevice *device, const BufferDesc &desc) : MetalBuffer(device, desc) {
}

MetalAllocatedBuffer *MetalAllocatedBuffer::Create(MetalDevice *device, const BufferDesc &desc, bool upload) {
  if (!desc.size || desc.size > device->NativeDevice()->maxBufferLength())
    return nullptr;

  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto object = std::unique_ptr<MetalAllocatedBuffer>(new MetalAllocatedBuffer(device, desc));
  object->buffer_ = NS::TransferPtr(device->NativeDevice()->newBuffer(
      desc.size, upload ? MTL::ResourceStorageModeShared : MTL::ResourceStorageModePrivate));
  if (!object->buffer_) {
    ReportError(-1, "Metal buffer allocation failed.");
    return nullptr;
  }
  return object.release();
}

MTL::Buffer *MetalAllocatedBuffer::NativeBuffer() const {
  return buffer_.get();
}

bool MetalAllocatedBuffer::Write(const void *data, uint64_t size) {
  if (!data || size > desc_.size || buffer_->storageMode() != MTL::StorageModeShared)
    return false;
  std::memcpy(buffer_->contents(), data, static_cast<size_t>(size));
  return true;
}
}  // namespace grassland::rhi::backend::metal
