#pragma once
#include <array>
#include <chrono>
#include <deque>
#include <mutex>
#include <unordered_map>

#include "grassland/rhi/backend/backend_util.h"

namespace grassland::rhi::backend {
enum TemporalBufferUsage { TEMPORAL_BUFFER_UPLOAD, TEMPORAL_BUFFER_DOWNLOAD };

// Defined and explicitly instantiated in the .cpp for supported native backends.
// Pools own storage; intrusive leases borrow it until their final reference dies.
template <typename DeviceType, typename BufferType, typename LeaseType>
class TemporalBufferPool {
 public:
  using Lease = LeaseType;
  TemporalBufferPool(DeviceType *device, TemporalBufferUsage usage);
  ~TemporalBufferPool();
  ObjectPtr<Lease> Acquire(uint64_t size);
  std::pair<size_t, size_t> GetBufferCounts(uint64_t size) const;
  void Clear();

 private:
  // A dependent lease destructor cannot be named in a portable friend
  // declaration. Only the concrete lease is authorized to return storage.
  friend LeaseType;
  void ReleaseBuffer(BufferType *buffer, size_t level);
  DeviceType *device_;
  const TemporalBufferUsage usage_;
  mutable std::mutex mutex_;

  struct Bucket {
    void Record(size_t occupancy, std::chrono::steady_clock::time_point now);
    std::unordered_map<BufferType *, ObjectPtr<BufferType>> buffers;
    std::vector<BufferType *> available;
    size_t allocated{}, occupied{};
    std::deque<std::pair<size_t, std::chrono::steady_clock::time_point>> peaks;
  };

  // 64 KiB through 2^63 bytes, indexed by the requested power-of-two class.
  std::array<Bucket, 48> buckets_;
};
}  // namespace grassland::rhi::backend
