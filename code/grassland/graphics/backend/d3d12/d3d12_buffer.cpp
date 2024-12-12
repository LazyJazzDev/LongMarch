#include "grassland/graphics/backend/d3d12/d3d12_buffer.h"

namespace grassland::graphics::backend {

D3D12StaticBuffer::D3D12StaticBuffer(size_t size, D3D12Core *core)
    : core_(core) {
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
  core_->CommandQueue()->SingleTimeCommand(
      [&](ID3D12GraphicsCommandList *command_list) {
        d3d12::CopyBuffer(command_list, buffer_.get(), new_buffer.get(),
                          std::min(buffer_->Size(), new_size));
      });

  buffer_.reset();
  buffer_ = std::move(new_buffer);
}

void D3D12StaticBuffer::UploadData(const void *data,
                                   size_t size,
                                   size_t offset) {
  core_->WaitGPU();
  std::unique_ptr<d3d12::Buffer> staging_buffer;
  core_->Device()->CreateBuffer(size, D3D12_HEAP_TYPE_UPLOAD, &staging_buffer);
  std::memcpy(staging_buffer->Map(), data, size);
  staging_buffer->Unmap();
  core_->CommandQueue()->SingleTimeCommand(
      [&](ID3D12GraphicsCommandList *command_list) {
        void *mapped_data = staging_buffer->Map();
        std::memcpy(mapped_data, data, size);
        staging_buffer->Unmap();
        d3d12::CopyBuffer(command_list, staging_buffer.get(), buffer_.get(),
                          size, 0, offset);
      });
}

void D3D12StaticBuffer::DownloadData(void *data, size_t size, size_t offset) {
  core_->WaitGPU();
  std::unique_ptr<d3d12::Buffer> staging_buffer;
  core_->Device()->CreateBuffer(size, D3D12_HEAP_TYPE_READBACK,
                                &staging_buffer);
  core_->CommandQueue()->SingleTimeCommand(
      [&](ID3D12GraphicsCommandList *command_list) {
        d3d12::CopyBuffer(command_list, buffer_.get(), staging_buffer.get(),
                          size, offset);
      });
  std::memcpy(data, staging_buffer->Map(), size);
  staging_buffer->Unmap();
}

}  // namespace grassland::graphics::backend
