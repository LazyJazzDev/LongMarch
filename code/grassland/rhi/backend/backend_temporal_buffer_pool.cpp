#include "grassland/rhi/backend/backend_temporal_buffer_pool.h"
#if defined(LONGMARCH_D3D12_ENABLED)
#include "grassland/rhi/backend/d3d12/d3d12_device.h"
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
#include "grassland/rhi/backend/vulkan/vulkan_device.h"
#endif
#if defined(LONGMARCH_METAL_ENABLED)
#include "grassland/rhi/backend/metal/metal_device.h"
#endif
namespace grassland::rhi::backend {
template <typename DeviceType, typename BufferType, typename LeaseType>
TemporalBufferPool<DeviceType, BufferType, LeaseType>::TemporalBufferPool(DeviceType *device, TemporalBufferUsage usage)
    : device_(device),
      usage_(usage) {
}

template <typename DeviceType, typename BufferType, typename LeaseType>
TemporalBufferPool<DeviceType, BufferType, LeaseType>::~TemporalBufferPool() = default;

template <typename DeviceType, typename BufferType, typename LeaseType>
void TemporalBufferPool<DeviceType, BufferType, LeaseType>::Bucket::Record(size_t occupancy,
                                                                           std::chrono::steady_clock::time_point now) {
  // Allocate first: failure must not erase previously recorded demand.
  peaks.emplace_back(occupancy, now);
  while (peaks.size() > 1 && peaks[peaks.size() - 2].first <= occupancy) {
    peaks.pop_back();
    peaks.back() = {occupancy, now};
  }
  while (now - peaks.front().second > std::chrono::seconds(1))
    peaks.pop_front();
}

template <typename DeviceType, typename BufferType, typename LeaseType>
ObjectPtr<LeaseType> TemporalBufferPool<DeviceType, BufferType, LeaseType>::Acquire(uint64_t size) {
  uint64_t capacity = 65536;
  size_t level = 0;
  while (capacity < size && capacity <= UINT64_MAX / 2) {
    capacity *= 2;
    ++level;
  }
  if (capacity < size)
    return {};
  BufferType *buffer = nullptr;
  try {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto &bucket = buckets_[level];
      if (bucket.available.empty()) {
        // Reserve return-list space before committing a new pool-owned allocation.
        bucket.available.reserve(bucket.allocated + 1);
        auto storage = ObjectPtr<BufferType>::Adopt(BufferType::Create(
            device_, {capacity}, usage_ == TEMPORAL_BUFFER_UPLOAD, usage_ == TEMPORAL_BUFFER_DOWNLOAD));
        if (!storage)
          return {};
        auto pointer = storage.Get();
        bucket.buffers.emplace(pointer, std::move(storage));
        bucket.available.push_back(pointer);
        ++bucket.allocated;
      }
      bucket.Record(bucket.occupied + 1, std::chrono::steady_clock::now());
      buffer = bucket.available.back();
      bucket.available.pop_back();
      ++bucket.occupied;
    }
    // Construct outside the lock: failed construction returns the borrowed slot.
    return ObjectPtr<Lease>::Adopt(new Lease(this, buffer, level));
  } catch (const std::bad_alloc &) {
    if (buffer)
      ReleaseBuffer(buffer, level);
    throw;
  }
}

template <typename DeviceType, typename BufferType, typename LeaseType>
void TemporalBufferPool<DeviceType, BufferType, LeaseType>::ReleaseBuffer(BufferType *buffer, size_t level) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto &bucket = buckets_[level];
  --bucket.occupied;
  try {
    bucket.Record(bucket.occupied, std::chrono::steady_clock::now());
  } catch (const std::bad_alloc &) {
    bucket.buffers.erase(buffer);
    --bucket.allocated;
    return;
  }
  bucket.available.push_back(buffer);  // Capacity reserved by Acquire.
  const auto demand = bucket.peaks.front().first;
  while (bucket.allocated > demand && !bucket.available.empty()) {
    bucket.buffers.erase(bucket.available.back());
    bucket.available.pop_back();
    --bucket.allocated;
  }
}

template <typename DeviceType, typename BufferType, typename LeaseType>
std::pair<size_t, size_t> TemporalBufferPool<DeviceType, BufferType, LeaseType>::GetBufferCounts(uint64_t size) const {
  uint64_t capacity = 65536;
  size_t level = 0;
  while (capacity < size && capacity <= UINT64_MAX / 2) {
    capacity *= 2;
    ++level;
  }
  if (capacity < size)
    return {};
  std::lock_guard<std::mutex> lock(mutex_);
  return {buckets_[level].allocated, buckets_[level].occupied};
}

template <typename DeviceType, typename BufferType, typename LeaseType>
void TemporalBufferPool<DeviceType, BufferType, LeaseType>::Clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto &bucket : buckets_) {
    bucket.allocated -= bucket.available.size();
    for (auto buffer : bucket.available)
      bucket.buffers.erase(buffer);
    bucket.available.clear();
    bucket.peaks.clear();
  }
}
#if defined(LONGMARCH_D3D12_ENABLED)
template class TemporalBufferPool<d3d12::D3D12Device, d3d12::D3D12AllocatedBuffer, d3d12::D3D12TemporalBufferLease>;
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
template class TemporalBufferPool<vulkan::VulkanDevice,
                                  vulkan::VulkanAllocatedBuffer,
                                  vulkan::VulkanTemporalBufferLease>;
#endif
#if defined(LONGMARCH_METAL_ENABLED)
template class TemporalBufferPool<metal::MetalDevice, metal::MetalAllocatedBuffer, metal::MetalTemporalBufferLease>;
#endif
}  // namespace grassland::rhi::backend
