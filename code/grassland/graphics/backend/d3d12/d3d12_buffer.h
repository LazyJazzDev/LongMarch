#pragma once
#include "grassland/graphics/backend/d3d12/d3d12_core.h"
#include "grassland/graphics/backend/d3d12/d3d12_util.h"

namespace grassland::graphics::backend {

class D3D12Buffer : virtual public Buffer {
 public:
  virtual ~D3D12Buffer() = default;

  virtual d3d12::Buffer *Buffer() const = 0;
  virtual d3d12::Buffer *InstantBuffer() const = 0;
};

class D3D12StaticBuffer : public D3D12Buffer {
 public:
  D3D12StaticBuffer(D3D12Core *core, size_t size);
  ~D3D12StaticBuffer() override;

  BufferType Type() const override;

  size_t Size() const override;

  void Resize(size_t new_size) override;

  void UploadData(const void *data, size_t size, size_t offset) override;

  void DownloadData(void *data, size_t size, size_t offset) override;

  d3d12::Buffer *Buffer() const override;

  d3d12::Buffer *InstantBuffer() const override;

 private:
  D3D12Core *core_;
  std::unique_ptr<d3d12::Buffer> buffer_;
};

class D3D12DynamicBuffer : public D3D12Buffer {
 public:
  D3D12DynamicBuffer(D3D12Core *core, size_t size);
  ~D3D12DynamicBuffer() override;

  BufferType Type() const override;

  size_t Size() const override;

  void Resize(size_t new_size) override;

  void UploadData(const void *data, size_t size, size_t offset) override;

  void DownloadData(void *data, size_t size, size_t offset) override;

  d3d12::Buffer *Buffer() const override;

  d3d12::Buffer *InstantBuffer() const override;

  void TransferData(ID3D12GraphicsCommandList *command_list);

 private:
  D3D12Core *core_;
  std::vector<std::unique_ptr<d3d12::Buffer>> buffers_;
  std::unique_ptr<d3d12::Buffer> staging_buffer_;
};

#if defined(LONGMARCH_CUDA_RUNTIME)
class D3D12CUDABuffer : public D3D12Buffer, public CUDABuffer {
 public:
  D3D12CUDABuffer(D3D12Core *core, size_t size);
  ~D3D12CUDABuffer() override;

  BufferType Type() const override;

  size_t Size() const override;

  void Resize(size_t new_size) override;

  void UploadData(const void *data, size_t size, size_t offset) override;

  void DownloadData(void *data, size_t size, size_t offset) override;

  d3d12::Buffer *Buffer() const override;

  d3d12::Buffer *InstantBuffer() const override;

  void GetCUDAMemoryPointer(void **ptr) override;

 private:
  D3D12Core *core_;
  std::unique_ptr<d3d12::Buffer> buffer_;
  cudaExternalMemory_t cuda_memory_;
};
#endif

}  // namespace grassland::graphics::backend
