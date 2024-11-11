#pragma once
#include "grassland/d3d12/device.h"

namespace grassland::d3d12 {

class Buffer {
 public:
  Buffer(const ComPtr<ID3D12Resource> &buffer);

  size_t Size() const {
    return buffer_->GetDesc().Width;
  }

  ID3D12Resource *Handle() const {
    return buffer_.Get();
  }

  void *Map() const;
  void Unmap() const;

  void UploadData(const void *data, size_t size, size_t offset = 0);
  void DownloadData(void *data, size_t size, size_t offset = 0);

 private:
  ComPtr<ID3D12Resource> buffer_;
};

}  // namespace grassland::d3d12
