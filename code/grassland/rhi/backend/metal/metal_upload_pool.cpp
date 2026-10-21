#include "grassland/rhi/backend/metal/metal_upload_pool.h"

#include "grassland/rhi/backend/metal/metal_device.h"

namespace grassland::rhi::backend::metal {
MetalUploadPool::Lease::Lease(MetalUploadPool *pool, MetalAllocatedBuffer *buffer, size_t level)
    : MetalBuffer(static_cast<MetalDevice *>(buffer->Owner()), buffer->GetDesc()),
      pool_(pool),
      buffer_(buffer),
      level_(level) {
}

MetalUploadPool::Lease::~Lease() {
  pool_->ReleaseBuffer(buffer_, level_);
}

BufferDesc MetalUploadPool::Lease::GetDesc() const {
  return buffer_->GetDesc();
}

MTL::Buffer *MetalUploadPool::Lease::NativeBuffer() const {
  return buffer_->NativeBuffer();
}

bool MetalUploadPool::Lease::Write(const void *data, uint64_t size) {
  return buffer_->Write(data, size);
}

MetalUploadPool::MetalUploadPool(MetalDevice *device) : device_(device) {
}

void MetalUploadPool::Bucket::Record(size_t occupancy, std::chrono::steady_clock::time_point now) {
  // Allocate first: failure must not erase previously recorded demand.
  peaks.emplace_back(occupancy, now);
  while (peaks.size() > 1 && peaks[peaks.size() - 2].first <= occupancy) {
    peaks.pop_back();
    peaks.back() = {occupancy, now};
  }
  while (now - peaks.front().second > std::chrono::seconds(1))
    peaks.pop_front();
}

ObjectPtr<MetalUploadPool::Lease> MetalUploadPool::Acquire(uint64_t size) {
  uint64_t capacity = 65536;
  size_t level = 0;
  while (capacity < size && capacity <= UINT64_MAX / 2) {
    capacity *= 2;
    ++level;
  }
  if (capacity < size)
    return {};
  MetalAllocatedBuffer *buffer = nullptr;
  try {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto &bucket = buckets_[level];
      if (bucket.available.empty()) {
        // Reserve return-list space before committing a new pool-owned allocation.
        bucket.available.reserve(bucket.allocated + 1);
        auto storage = ObjectPtr<MetalAllocatedBuffer>::Adopt(MetalAllocatedBuffer::Create(device_, {capacity}, true));
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

void MetalUploadPool::ReleaseBuffer(MetalAllocatedBuffer *buffer, size_t level) {
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

std::pair<size_t, size_t> MetalUploadPool::GetBufferCounts(uint64_t size) const {
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

void MetalUploadPool::Clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto &bucket : buckets_) {
    bucket.allocated -= bucket.available.size();
    for (auto buffer : bucket.available)
      bucket.buffers.erase(buffer);
    bucket.available.clear();
    bucket.peaks.clear();
  }
}
}  // namespace grassland::rhi::backend::metal
