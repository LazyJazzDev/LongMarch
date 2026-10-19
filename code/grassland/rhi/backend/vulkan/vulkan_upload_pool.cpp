#include "grassland/rhi/backend/vulkan/vulkan_upload_pool.h"

#include "grassland/rhi/backend/vulkan/vulkan_device.h"

namespace grassland::rhi::backend::vulkan {
VulkanUploadPool::VulkanUploadPool(VulkanDevice *device) : device_(device) {
}

void VulkanUploadPool::Bucket::Record(size_t occupancy, std::chrono::steady_clock::time_point now) {
  // Allocate first: failure must not erase previously recorded demand.
  peaks.emplace_back(occupancy, now);
  while (peaks.size() > 1 && peaks[peaks.size() - 2].first <= occupancy) {
    peaks.pop_back();
    peaks.back() = {occupancy, now};
  }
  while (now - peaks.front().second > std::chrono::seconds(1))
    peaks.pop_front();
}

ObjectPtr<VulkanAllocatedBuffer> VulkanUploadPool::Acquire(uint64_t size) {
  uint64_t capacity = 65536;
  size_t level = 0;
  while (capacity < size && capacity <= UINT64_MAX / 2) {
    capacity *= 2;
    ++level;
  }
  if (capacity < size)
    return {};
  std::lock_guard<std::mutex> lock(mutex_);
  auto &bucket = buckets_[level];
  try {
    if (!bucket.available.empty()) {
      bucket.Record(bucket.occupied + 1, std::chrono::steady_clock::now());
      auto result = std::move(bucket.available.back());
      bucket.available.pop_back();
      ++bucket.occupied;
      return result;
    }
    auto result = ObjectPtr<VulkanAllocatedBuffer>::Adopt(new VulkanAllocatedBuffer(device_, {capacity}));
    if (!result->Initialize(true))
      return {};
    bucket.Record(bucket.occupied + 1, std::chrono::steady_clock::now());
    ++bucket.allocated;
    ++bucket.occupied;
    return result;
  } catch (const std::bad_alloc &) {
    return {};
  }
}

void VulkanUploadPool::Release(ObjectPtr<VulkanAllocatedBuffer> buffer) {
  size_t level = 0;
  for (auto capacity = buffer->GetDesc().size; capacity > 65536; capacity /= 2)
    ++level;
  std::lock_guard<std::mutex> lock(mutex_);
  auto &bucket = buckets_[level];
  --bucket.occupied;
  try {
    bucket.Record(bucket.occupied, std::chrono::steady_clock::now());
    bucket.available.push_back(std::move(buffer));
  } catch (const std::bad_alloc &) {
    --bucket.allocated;
    return;
  }
  const auto demand = bucket.peaks.front().first;
  while (bucket.allocated > demand && !bucket.available.empty()) {
    bucket.available.pop_back();
    --bucket.allocated;
  }
}

void VulkanUploadPool::Clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto &bucket : buckets_) {
    bucket.allocated -= bucket.available.size();
    bucket.available.clear();
    bucket.peaks.clear();
  }
}
}  // namespace grassland::rhi::backend::vulkan
