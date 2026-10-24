#pragma once
#include <algorithm>
#include <mutex>
#include <vector>

#include "grassland/rhi/backend/backend_util.h"

namespace grassland::rhi::backend {
// Device-owned storage; each shared lease returns one slot on final release.
// Commands and download tokens share a lease, never the ownership of the pool.
template <typename Buffer>
class DownloadPool {
 public:
  template <typename Factory>
  std::shared_ptr<Buffer> Acquire(uint64_t size, Factory create) {
    uint64_t capacity = 65536;
    while (capacity < size && capacity <= UINT64_MAX / 2)
      capacity *= 2;
    if (!size || capacity < size)
      return {};
    std::lock_guard<std::mutex> lock(mutex_);
    Slot *slot = nullptr;
    for (auto &candidate : slots_)
      if (!candidate->occupied && candidate->storage->GetDesc().size == capacity) {
        slot = candidate.get();
        break;
      }
    if (!slot) {
      auto storage = ObjectPtr<Buffer>::Adopt(create(capacity));
      if (!storage)
        return {};
      auto entry = std::make_unique<Slot>();
      entry->storage = std::move(storage);
      slot = entry.get();
      slots_.push_back(std::move(entry));
    }
    // The atomic return path also handles control-block allocation failure.
    slot->occupied = true;
    return std::shared_ptr<Buffer>(slot->storage.Get(),
                                   [slot](Buffer *) { slot->occupied.store(false, std::memory_order_release); });
  }

  void Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    slots_.erase(std::remove_if(slots_.begin(), slots_.end(), [](const auto &s) { return !s->occupied.load(); }),
                 slots_.end());
  }

 private:
  struct Slot {
    ObjectPtr<Buffer> storage;
    std::atomic<bool> occupied{false};
  };

  std::mutex mutex_;
  std::vector<std::unique_ptr<Slot>> slots_;
};
}  // namespace grassland::rhi::backend
