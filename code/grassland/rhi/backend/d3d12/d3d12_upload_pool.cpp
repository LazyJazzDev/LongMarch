#include "grassland/rhi/backend/d3d12/d3d12_upload_pool.h"

#include "grassland/rhi/backend/d3d12/d3d12_device.h"

namespace grassland::rhi::backend::d3d12 {
D3D12UploadPool::D3D12UploadPool(D3D12Device *device) : device_(device) {
}

ObjectPtr<D3D12AllocatedBuffer> D3D12UploadPool::Acquire(uint64_t size) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto it = available_.begin(); it != available_.end(); ++it) {
    if ((*it)->GetDesc().size >= size) {
      auto result = std::move(*it);
      available_.erase(it);
      return result;
    }
  }
  uint64_t capacity = 65536;
  while (capacity < size && capacity <= UINT64_MAX / 2)
    capacity *= 2;
  if (capacity < size)
    return {};
  auto result = ObjectPtr<D3D12AllocatedBuffer>::Adopt(new D3D12AllocatedBuffer(device_, {capacity}));
  if (!result->Initialize(true))
    return {};
  return result;
}

void D3D12UploadPool::Release(ObjectPtr<D3D12AllocatedBuffer> buffer) {
  try {
    std::lock_guard<std::mutex> lock(mutex_);
    available_.push_back(std::move(buffer));
  } catch (const std::bad_alloc &) {
  }
}

void D3D12UploadPool::Clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  available_.clear();
}
}  // namespace grassland::rhi::backend::d3d12
