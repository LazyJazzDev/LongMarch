#include "grassland/rhi/backend/metal/metal_scratch_pool.h"

#include "grassland/rhi/backend/metal/metal_device.h"

namespace grassland::rhi::backend::metal {
MetalScratchPool::MetalScratchPool(MetalDevice *device) : device_(device) {
}

ObjectPtr<MetalAllocatedBuffer> MetalScratchPool::Acquire(uint64_t size) {
  if (size > (uint64_t(1) << 63))
    return {};
  std::lock_guard<std::mutex> lock(mutex_);
  const auto now = std::chrono::steady_clock::now();
  // Append before removing history so allocation failure preserves previous demand.
  peaks_.emplace_back(size, now);
  while (peaks_.size() > 1 && peaks_[peaks_.size() - 2].first <= size) {
    peaks_.pop_back();
    peaks_.back() = {size, now};
  }
  while (now - peaks_.front().second > std::chrono::seconds(1))
    peaks_.pop_front();

  const auto demand = peaks_.front().first;
  const auto current = buffer_ ? buffer_->GetDesc().size : 0;
  if (!buffer_ || size > current || demand < current / 2) {
    uint64_t capacity = 65536;
    while (capacity < demand)
      capacity *= 2;
    if (capacity != current) {
      auto storage = ObjectPtr<MetalAllocatedBuffer>::Adopt(MetalAllocatedBuffer::Create(device_, {capacity}, false));
      if (!storage) {
        // Shrinking is optional; a failed growth must not return undersized storage.
        if (size <= current)
          return buffer_;
        return {};
      }
      buffer_ = std::move(storage);
    }
  }
  return buffer_;
}

void MetalScratchPool::Clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  buffer_.Reset();
  peaks_.clear();
}
}  // namespace grassland::rhi::backend::metal
