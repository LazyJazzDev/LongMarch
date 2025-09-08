#pragma once
#include "grassland/d3d12/device.h"

namespace CD::d3d12 {

class Buffer {
 public:
  Buffer(const ComPtr<ID3D12Resource> &buffer, size_t allocated_size);

  size_t Size() const {
    return allocated_size_;
  }

  ID3D12Resource *Handle() const {
    return buffer_.Get();
  }

  void *Map() const;
  void Unmap() const;

 private:
  ComPtr<ID3D12Resource> buffer_;
  size_t allocated_size_;
};

void CopyBuffer(ID3D12GraphicsCommandList *command_list,
                Buffer *src_buffer,
                Buffer *dst_buffer,
                size_t size,
                size_t src_offset = 0,
                size_t dst_offset = 0);

}  // namespace CD::d3d12
