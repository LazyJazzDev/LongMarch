#pragma once
#include <array>
#include <chrono>
#include <deque>
#include <mutex>

#include "grassland/rhi/backend/vulkan/vulkan_allocated_buffer.h"

namespace grassland::rhi::backend::vulkan {
class VulkanUploadPool {
 public:
  explicit VulkanUploadPool(VulkanDevice *device);
  ObjectPtr<VulkanAllocatedBuffer> Acquire(uint64_t size);
  void Release(ObjectPtr<VulkanAllocatedBuffer> buffer);
  void Clear();

 private:
  VulkanDevice *device_;
  std::mutex mutex_;

  struct Bucket {
    void Record(size_t occupancy, std::chrono::steady_clock::time_point now);
    std::vector<ObjectPtr<VulkanAllocatedBuffer>> available;
    size_t allocated{}, occupied{};
    std::deque<std::pair<size_t, std::chrono::steady_clock::time_point>> peaks;
  };

  // 64 KiB through 2^63 bytes, indexed directly by the requested power-of-two class.
  std::array<Bucket, 48> buckets_;
};
}  // namespace grassland::rhi::backend::vulkan
