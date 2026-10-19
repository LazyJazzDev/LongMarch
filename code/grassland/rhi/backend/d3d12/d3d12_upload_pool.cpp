#include "grassland/rhi/backend/d3d12/d3d12_upload_pool.h"

#include "grassland/rhi/backend/d3d12/d3d12_device.h"

namespace grassland::rhi::backend::d3d12 {
D3D12UploadPool::D3D12UploadPool(D3D12Device *device) : device_(device) {
}

void D3D12UploadPool::Bucket::Record(size_t occupancy, std::chrono::steady_clock::time_point now) {
  // Allocate first: failure must not erase previously recorded demand.
  peaks.emplace_back(occupancy, now);
  while (peaks.size() > 1 && peaks[peaks.size() - 2].first <= occupancy) {
    peaks.pop_back();
    peaks.back() = {occupancy, now};
  }
  while (now - peaks.front().second > std::chrono::seconds(1))
    peaks.pop_front();
}

ObjectPtr<D3D12UploadBufferLease> D3D12UploadPool::Acquire(uint64_t size) {
  uint64_t capacity = 65536;
  size_t level = 0;
  while (capacity < size && capacity <= UINT64_MAX / 2) {
    capacity *= 2;
    ++level;
  }
  if (capacity < size)
    return {};
  D3D12AllocatedBuffer *buffer = nullptr;
  try {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto &bucket = buckets_[level];
      if (bucket.available.empty()) {
        // Reserve return-list space before committing a new pool-owned allocation.
        bucket.available.reserve(bucket.allocated + 1);
        auto storage = ObjectPtr<D3D12AllocatedBuffer>::Adopt(new D3D12AllocatedBuffer(device_, {capacity}));
        if (!storage->Initialize(true))
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
    return ObjectPtr<D3D12UploadBufferLease>::Adopt(new D3D12UploadBufferLease(this, buffer));
  } catch (const std::bad_alloc &) {
    if (buffer)
      ReleaseBuffer(buffer);
    return {};
  }
}

void D3D12UploadPool::ReleaseBuffer(D3D12AllocatedBuffer *buffer) {
  size_t level = 0;
  for (auto capacity = buffer->GetDesc().size; capacity > 65536; capacity /= 2)
    ++level;
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

std::pair<size_t, size_t> D3D12UploadPool::GetBufferCounts(uint64_t size) const {
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

void D3D12UploadPool::Clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto &bucket : buckets_) {
    bucket.allocated -= bucket.available.size();
    for (auto buffer : bucket.available)
      bucket.buffers.erase(buffer);
    bucket.available.clear();
    bucket.peaks.clear();
  }
}
}  // namespace grassland::rhi::backend::d3d12
