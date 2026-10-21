#include "grassland/rhi/backend/d3d12/d3d12_scratch_pool.h"

#include "grassland/rhi/backend/d3d12/d3d12_device.h"

namespace grassland::rhi::backend::d3d12 {
D3D12ScratchPool::Lease::Lease(D3D12ScratchPool *pool, D3D12AllocatedBuffer *buffer, size_t level)
    : D3D12Buffer(static_cast<D3D12Device *>(buffer->Owner()), buffer->GetDesc()),
      pool_(pool),
      buffer_(buffer),
      level_(level) {
}

D3D12ScratchPool::Lease::~Lease() {
  pool_->ReleaseBuffer(buffer_, level_);
}

BufferDesc D3D12ScratchPool::Lease::GetDesc() const {
  return buffer_->GetDesc();
}

ID3D12Resource *D3D12ScratchPool::Lease::NativeBuffer() const {
  return buffer_->NativeBuffer();
}

D3D12_RESOURCE_STATES D3D12ScratchPool::Lease::DefaultState() const {
  return buffer_->DefaultState();
}

D3D12ScratchPool::D3D12ScratchPool(D3D12Device *device) : device_(device) {
}

void D3D12ScratchPool::Bucket::Record(size_t occupancy, std::chrono::steady_clock::time_point now) {
  // Allocate first: failure must not erase previously recorded demand.
  peaks.emplace_back(occupancy, now);
  while (peaks.size() > 1 && peaks[peaks.size() - 2].first <= occupancy) {
    peaks.pop_back();
    peaks.back() = {occupancy, now};
  }
  while (now - peaks.front().second > std::chrono::seconds(1))
    peaks.pop_front();
}

ObjectPtr<D3D12ScratchPool::Lease> D3D12ScratchPool::Acquire(uint64_t size) {
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
        auto storage = ObjectPtr<D3D12AllocatedBuffer>::Adopt(D3D12AllocatedBuffer::Create(device_, {capacity}, false));
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

void D3D12ScratchPool::ReleaseBuffer(D3D12AllocatedBuffer *buffer, size_t level) {
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

std::pair<size_t, size_t> D3D12ScratchPool::GetBufferCounts(uint64_t size) const {
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

void D3D12ScratchPool::Clear() {
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
