#include "grassland/graphics/backend/d3d12/d3d12_buffer.h"

namespace grassland::graphics::backend {

D3D12BufferRange::D3D12BufferRange(const BufferRange &range)
    : buffer(dynamic_cast<D3D12Buffer *>(range.buffer)), offset(range.offset), size(range.size) {
}

D3D12StaticBuffer::D3D12StaticBuffer(D3D12Core *core, size_t size) : core_(core) {
  core_->Device()->CreateBuffer(size, D3D12_HEAP_TYPE_DEFAULT, &buffer_);
}

D3D12StaticBuffer::~D3D12StaticBuffer() {
  buffer_.reset();
}

BufferType D3D12StaticBuffer::Type() const {
  return BUFFER_TYPE_STATIC;
}

size_t D3D12StaticBuffer::Size() const {
  return buffer_->Size();
}

void D3D12StaticBuffer::Resize(size_t new_size) {
  core_->WaitGPU();
  std::unique_ptr<d3d12::Buffer> new_buffer;
  core_->Device()->CreateBuffer(new_size, D3D12_HEAP_TYPE_DEFAULT, &new_buffer);
  core_->SingleTimeCommand([&](ID3D12GraphicsCommandList *command_list) {
    d3d12::CopyBuffer(command_list, buffer_.get(), new_buffer.get(), std::min(buffer_->Size(), new_size));
  });

  buffer_.reset();
  buffer_ = std::move(new_buffer);
}

void D3D12StaticBuffer::UploadData(const void *data, size_t size, size_t offset) {
  core_->WaitGPU();
  auto staging_buffer = core_->RequestUploadStagingBuffer(size);
  std::memcpy(staging_buffer->Map(), data, size);
  staging_buffer->Unmap();
  core_->SingleTimeCommand([&](ID3D12GraphicsCommandList *command_list) {
    void *mapped_data = staging_buffer->Map();
    std::memcpy(mapped_data, data, size);
    staging_buffer->Unmap();
    d3d12::CopyBuffer(command_list, staging_buffer, buffer_.get(), size, 0, offset);
  });
}

void D3D12StaticBuffer::DownloadData(void *data, size_t size, size_t offset) {
  core_->WaitGPU();
  auto staging_buffer = core_->RequestDownloadStagingBuffer(size);
  core_->SingleTimeCommand([&](ID3D12GraphicsCommandList *command_list) {
    d3d12::CopyBuffer(command_list, buffer_.get(), staging_buffer, size, offset);
  });
  std::memcpy(data, staging_buffer->Map(), size);
  staging_buffer->Unmap();
}

d3d12::Buffer *D3D12StaticBuffer::Buffer() const {
  return buffer_.get();
}

d3d12::Buffer *D3D12StaticBuffer::InstantBuffer() const {
  return buffer_.get();
}

D3D12DynamicBuffer::D3D12DynamicBuffer(D3D12Core *core, size_t size) : core_(core) {
  buffers_.resize(core_->FramesInFlight());
  for (size_t i = 0; i < buffers_.size(); ++i) {
    core_->Device()->CreateBuffer(size, D3D12_HEAP_TYPE_DEFAULT, &buffers_[i]);
  }
  core_->Device()->CreateBuffer(size, D3D12_HEAP_TYPE_UPLOAD, &staging_buffer_);
}

D3D12DynamicBuffer::~D3D12DynamicBuffer() {
  buffers_.clear();
  staging_buffer_.reset();
}

BufferType D3D12DynamicBuffer::Type() const {
  return BUFFER_TYPE_DYNAMIC;
}

size_t D3D12DynamicBuffer::Size() const {
  return staging_buffer_->Size();
}

void D3D12DynamicBuffer::Resize(size_t new_size) {
  std::unique_ptr<d3d12::Buffer> new_buffer;
  core_->Device()->CreateBuffer(new_size, D3D12_HEAP_TYPE_UPLOAD, &new_buffer);

  std::memcpy(new_buffer->Map(), staging_buffer_->Map(), std::min(new_size, Size()));
  new_buffer->Unmap();
  staging_buffer_->Unmap();

  staging_buffer_.reset();
  staging_buffer_ = std::move(new_buffer);
}

void D3D12DynamicBuffer::UploadData(const void *data, size_t size, size_t offset) {
  std::memcpy(static_cast<uint8_t *>(staging_buffer_->Map()) + offset, data, size);
  staging_buffer_->Unmap();
}

void D3D12DynamicBuffer::DownloadData(void *data, size_t size, size_t offset) {
  std::memcpy(data, static_cast<uint8_t *>(staging_buffer_->Map()) + offset, size);
  staging_buffer_->Unmap();
}

d3d12::Buffer *D3D12DynamicBuffer::Buffer() const {
  return buffers_[core_->CurrentFrame()].get();
}

d3d12::Buffer *D3D12DynamicBuffer::InstantBuffer() const {
  return staging_buffer_.get();
}

void D3D12DynamicBuffer::TransferData(ID3D12GraphicsCommandList *command_list) {
  if (buffers_[core_->CurrentFrame()]->Size() != staging_buffer_->Size()) {
    buffers_[core_->CurrentFrame()].reset();
    core_->Device()->CreateBuffer(staging_buffer_->Size(), D3D12_HEAP_TYPE_DEFAULT, &buffers_[core_->CurrentFrame()]);
  }

  command_list->CopyBufferRegion(buffers_[core_->CurrentFrame()]->Handle(), 0, staging_buffer_->Handle(), 0,
                                 staging_buffer_->Size());
}

#if defined(LONGMARCH_CUDA_RUNTIME)
D3D12CUDABuffer::D3D12CUDABuffer(D3D12Core *core, size_t size) : core_(core) {
  core_->Device()->CreateBuffer(size, D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_SHARED,
                                D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                &buffer_);
  core_->ImportCudaExternalMemory(cuda_memory_, buffer_.get());
}

D3D12CUDABuffer::~D3D12CUDABuffer() {
  cudaDestroyExternalMemory(cuda_memory_);
  buffer_.reset();
}

BufferType D3D12CUDABuffer::Type() const {
  return BUFFER_TYPE_STATIC;
}

size_t D3D12CUDABuffer::Size() const {
  return buffer_->Size();
}

void D3D12CUDABuffer::Resize(size_t new_size) {
  core_->WaitGPU();
  std::unique_ptr<d3d12::Buffer> new_buffer;
  core_->Device()->CreateBuffer(new_size, D3D12_HEAP_TYPE_DEFAULT, &new_buffer);
  core_->SingleTimeCommand([&](ID3D12GraphicsCommandList *command_list) {
    d3d12::CopyBuffer(command_list, buffer_.get(), new_buffer.get(), std::min(buffer_->Size(), new_size));
  });
  cudaDestroyExternalMemory(cuda_memory_);
  buffer_.reset();

  buffer_ = std::move(new_buffer);
  core_->ImportCudaExternalMemory(cuda_memory_, buffer_.get());
}

void D3D12CUDABuffer::UploadData(const void *data, size_t size, size_t offset) {
  core_->WaitGPU();
  auto staging_buffer = core_->RequestUploadStagingBuffer(size);
  std::memcpy(staging_buffer->Map(), data, size);
  staging_buffer->Unmap();
  core_->SingleTimeCommand([&](ID3D12GraphicsCommandList *command_list) {
    void *mapped_data = staging_buffer->Map();
    std::memcpy(mapped_data, data, size);
    staging_buffer->Unmap();
    d3d12::CopyBuffer(command_list, staging_buffer, buffer_.get(), size, 0, offset);
  });
}

void D3D12CUDABuffer::DownloadData(void *data, size_t size, size_t offset) {
  core_->WaitGPU();
  auto staging_buffer = core_->RequestDownloadStagingBuffer(size);
  core_->Device()->CreateBuffer(size, D3D12_HEAP_TYPE_READBACK, &staging_buffer);
  core_->SingleTimeCommand([&](ID3D12GraphicsCommandList *command_list) {
    d3d12::CopyBuffer(command_list, buffer_.get(), staging_buffer, size, offset);
  });
  std::memcpy(data, staging_buffer->Map(), size);
  staging_buffer->Unmap();
}

d3d12::Buffer *D3D12CUDABuffer::Buffer() const {
  return buffer_.get();
}

d3d12::Buffer *D3D12CUDABuffer::InstantBuffer() const {
  return buffer_.get();
}

void D3D12CUDABuffer::GetCUDAMemoryPointer(void **ptr) {
  cudaExternalMemoryBufferDesc externalMemBufferDesc = {};
  externalMemBufferDesc.offset = 0;
  externalMemBufferDesc.size = buffer_->Size();
  externalMemBufferDesc.flags = 0;
  cudaExternalMemoryGetMappedBuffer(ptr, cuda_memory_, &externalMemBufferDesc);
}
#endif

}  // namespace grassland::graphics::backend
