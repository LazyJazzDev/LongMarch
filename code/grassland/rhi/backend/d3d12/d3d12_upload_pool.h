#pragma once
#include <mutex>

#include "grassland/rhi/backend/d3d12/d3d12_allocated_buffer.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12UploadPool {
 public:
  explicit D3D12UploadPool(D3D12Device *device);
  ObjectPtr<D3D12AllocatedBuffer> Acquire(uint64_t size);
  void Release(ObjectPtr<D3D12AllocatedBuffer> buffer);
  void Clear();

 private:
  D3D12Device *device_;
  std::mutex mutex_;
  std::vector<ObjectPtr<D3D12AllocatedBuffer>> available_;
};
}  // namespace grassland::rhi::backend::d3d12
