#pragma once
#include "grassland/graphics/backend/d3d12/d3d12_core.h"
#include "grassland/graphics/backend/d3d12/d3d12_util.h"

namespace grassland::graphics::backend {

class D3D12Buffer : public Buffer {};

class D3D12StaticBuffer : public D3D12Buffer {
 public:
  D3D12StaticBuffer(D3D12Core *core, size_t size);
  ~D3D12StaticBuffer() override;

  BufferType Type() const override;

  size_t Size() const override;

  void Resize(size_t new_size) override;

  void UploadData(const void *data, size_t size, size_t offset) override;

  void DownloadData(void *data, size_t size, size_t offset) override;

 private:
  D3D12Core *core_;
  std::unique_ptr<d3d12::Buffer> buffer_;
};

}  // namespace grassland::graphics::backend
