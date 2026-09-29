#include "grassland/graphics/backend/d3d12/d3d12_buffer.h"

namespace grassland::graphics::backend {

namespace {
void *MapBuffer(ID3D12Resource *buffer) {
  void *data = nullptr;
  d3d12::ThrowIfFailed(buffer->Map(0, nullptr, &data), "Failed to map buffer");
  return data;
}

void CopyNativeBuffer(ID3D12GraphicsCommandList *command_list,
                      ID3D12Resource *src,
                      ID3D12Resource *dst,
                      size_t size,
                      size_t src_offset = 0,
                      size_t dst_offset = 0,
                      D3D12_RESOURCE_STATES dst_original_state = D3D12_RESOURCE_STATE_GENERIC_READ) {
  if (dst_original_state != D3D12_RESOURCE_STATE_COPY_DEST) {
    auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(dst, dst_original_state, D3D12_RESOURCE_STATE_COPY_DEST);
    command_list->ResourceBarrier(1, &barrier);
  }
  command_list->CopyBufferRegion(dst, dst_offset, src, src_offset, size);
  if (dst_original_state != D3D12_RESOURCE_STATE_COPY_DEST) {
    auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(dst, D3D12_RESOURCE_STATE_COPY_DEST, dst_original_state);
    command_list->ResourceBarrier(1, &barrier);
  }
}
}  // namespace

D3D12BufferRange::D3D12BufferRange(const BufferRange &range)
    : buffer(dynamic_cast<D3D12Buffer *>(range.buffer)),
      offset(range.offset),
      size(range.size) {
}

D3D12StaticBuffer::D3D12StaticBuffer(D3D12Core *core, size_t size) : core_(core) {
  buffer_ = CreateNativeBuffer(core_->Device(), size, D3D12_HEAP_TYPE_DEFAULT);
}

D3D12StaticBuffer::~D3D12StaticBuffer() {
  buffer_.Reset();
}

BufferType D3D12StaticBuffer::Type() const {
  return BUFFER_TYPE_STATIC;
}

size_t D3D12StaticBuffer::Size() const {
  return buffer_->GetDesc().Width;
}

void D3D12StaticBuffer::Resize(size_t new_size) {
  core_->WaitGPU();
  ComPtr<ID3D12Resource> new_buffer;
  new_buffer = CreateNativeBuffer(core_->Device(), new_size, D3D12_HEAP_TYPE_DEFAULT);
  core_->SingleTimeCommand([&](ID3D12GraphicsCommandList *command_list) {
    CopyNativeBuffer(command_list, buffer_.Get(), new_buffer.Get(), std::min(buffer_->GetDesc().Width, new_size));
  });

  buffer_.Reset();
  buffer_ = std::move(new_buffer);
}

void D3D12StaticBuffer::UploadData(const void *data, size_t size, size_t offset) {
  core_->WaitGPU();
  auto staging_buffer = core_->RequestUploadStagingBuffer(size);
  std::memcpy(MapBuffer(staging_buffer), data, size);
  staging_buffer->Unmap(0, nullptr);
  core_->SingleTimeCommand([&](ID3D12GraphicsCommandList *command_list) {
    CopyNativeBuffer(command_list, staging_buffer, buffer_.Get(), size, 0, offset);
  });
}

void D3D12StaticBuffer::DownloadData(void *data, size_t size, size_t offset) {
  core_->WaitGPU();
  auto staging_buffer = core_->RequestDownloadStagingBuffer(size);
  core_->SingleTimeCommand([&](ID3D12GraphicsCommandList *command_list) {
    CopyNativeBuffer(command_list, buffer_.Get(), staging_buffer, size, offset, 0, D3D12_RESOURCE_STATE_COPY_DEST);
  });
  std::memcpy(data, MapBuffer(staging_buffer), size);
  staging_buffer->Unmap(0, nullptr);
}

ID3D12Resource *D3D12StaticBuffer::Buffer() const {
  return buffer_.Get();
}

ID3D12Resource *D3D12StaticBuffer::InstantBuffer() const {
  return buffer_.Get();
}

D3D12DynamicBuffer::D3D12DynamicBuffer(D3D12Core *core, size_t size) : core_(core) {
  buffers_.resize(core_->FramesInFlight());
  for (size_t i = 0; i < buffers_.size(); ++i) {
    buffers_[i] = CreateNativeBuffer(core_->Device(), size, D3D12_HEAP_TYPE_DEFAULT);
  }
  staging_buffer_ = CreateNativeBuffer(core_->Device(), size, D3D12_HEAP_TYPE_UPLOAD);
}

D3D12DynamicBuffer::~D3D12DynamicBuffer() {
  buffers_.clear();
  staging_buffer_.Reset();
}

BufferType D3D12DynamicBuffer::Type() const {
  return BUFFER_TYPE_DYNAMIC;
}

size_t D3D12DynamicBuffer::Size() const {
  return staging_buffer_->GetDesc().Width;
}

void D3D12DynamicBuffer::Resize(size_t new_size) {
  ComPtr<ID3D12Resource> new_buffer;
  new_buffer = CreateNativeBuffer(core_->Device(), new_size, D3D12_HEAP_TYPE_UPLOAD);

  std::memcpy(MapBuffer(new_buffer.Get()), MapBuffer(staging_buffer_.Get()), std::min(new_size, Size()));
  new_buffer->Unmap(0, nullptr);
  staging_buffer_->Unmap(0, nullptr);

  staging_buffer_.Reset();
  staging_buffer_ = std::move(new_buffer);
}

void D3D12DynamicBuffer::UploadData(const void *data, size_t size, size_t offset) {
  std::memcpy(static_cast<uint8_t *>(MapBuffer(staging_buffer_.Get())) + offset, data, size);
  staging_buffer_->Unmap(0, nullptr);
}

void D3D12DynamicBuffer::DownloadData(void *data, size_t size, size_t offset) {
  std::memcpy(data, static_cast<uint8_t *>(MapBuffer(staging_buffer_.Get())) + offset, size);
  staging_buffer_->Unmap(0, nullptr);
}

ID3D12Resource *D3D12DynamicBuffer::Buffer() const {
  return buffers_[core_->CurrentFrame()].Get();
}

ID3D12Resource *D3D12DynamicBuffer::InstantBuffer() const {
  return staging_buffer_.Get();
}

void D3D12DynamicBuffer::TransferData(ID3D12GraphicsCommandList *command_list) {
  if (buffers_[core_->CurrentFrame()]->GetDesc().Width != staging_buffer_->GetDesc().Width) {
    buffers_[core_->CurrentFrame()].Reset();
    buffers_[core_->CurrentFrame()] =
        CreateNativeBuffer(core_->Device(), staging_buffer_->GetDesc().Width, D3D12_HEAP_TYPE_DEFAULT);
  }
  CopyNativeBuffer(command_list, staging_buffer_.Get(), buffers_[core_->CurrentFrame()].Get(),
                   staging_buffer_->GetDesc().Width, 0, 0);
}

#if defined(LONGMARCH_CUDA_RUNTIME)
D3D12CUDABuffer::D3D12CUDABuffer(D3D12Core *core, size_t size) : core_(core) {
  buffer_ = CreateNativeBuffer(core_->Device(), size, D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_SHARED,
                               D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
  core_->ImportCudaExternalMemory(cuda_memory_, buffer_.Get());
}

D3D12CUDABuffer::~D3D12CUDABuffer() {
  cudaDestroyExternalMemory(cuda_memory_);
  buffer_.Reset();
}

BufferType D3D12CUDABuffer::Type() const {
  return BUFFER_TYPE_STATIC;
}

size_t D3D12CUDABuffer::Size() const {
  return buffer_->GetDesc().Width;
}

void D3D12CUDABuffer::Resize(size_t new_size) {
  core_->WaitGPU();
  ComPtr<ID3D12Resource> new_buffer;
  new_buffer = CreateNativeBuffer(core_->Device(), new_size, D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_SHARED,
                                  D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
  core_->SingleTimeCommand([&](ID3D12GraphicsCommandList *command_list) {
    CopyNativeBuffer(command_list, buffer_.Get(), new_buffer.Get(), std::min(buffer_->GetDesc().Width, new_size));
  });
  cudaDestroyExternalMemory(cuda_memory_);
  buffer_.Reset();

  buffer_ = std::move(new_buffer);
  core_->ImportCudaExternalMemory(cuda_memory_, buffer_.Get());
}

void D3D12CUDABuffer::UploadData(const void *data, size_t size, size_t offset) {
  core_->WaitGPU();
  auto staging_buffer = core_->RequestUploadStagingBuffer(size);
  std::memcpy(MapBuffer(staging_buffer), data, size);
  staging_buffer->Unmap(0, nullptr);
  core_->SingleTimeCommand([&](ID3D12GraphicsCommandList *command_list) {
    CopyNativeBuffer(command_list, staging_buffer, buffer_.Get(), size, 0, offset);
  });
}

void D3D12CUDABuffer::DownloadData(void *data, size_t size, size_t offset) {
  core_->WaitGPU();
  auto staging_buffer = core_->RequestDownloadStagingBuffer(size);
  core_->SingleTimeCommand([&](ID3D12GraphicsCommandList *command_list) {
    CopyNativeBuffer(command_list, buffer_.Get(), staging_buffer, size, offset);
  });
  std::memcpy(data, MapBuffer(staging_buffer), size);
  staging_buffer->Unmap(0, nullptr);
}

ID3D12Resource *D3D12CUDABuffer::Buffer() const {
  return buffer_.Get();
}

ID3D12Resource *D3D12CUDABuffer::InstantBuffer() const {
  return buffer_.Get();
}

void D3D12CUDABuffer::GetCUDAMemoryPointer(void **ptr) {
  cudaExternalMemoryBufferDesc externalMemBufferDesc = {};
  externalMemBufferDesc.offset = 0;
  externalMemBufferDesc.size = buffer_->GetDesc().Width;
  externalMemBufferDesc.flags = 0;
  cudaExternalMemoryGetMappedBuffer(ptr, cuda_memory_, &externalMemBufferDesc);
}
#endif

}  // namespace grassland::graphics::backend
