#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace grassland::graphics {

// Nonblocking backpressure: callers may retire work and record a fresh context.
class ResourcePoolExhausted : public std::runtime_error {
 public:
  explicit ResourcePoolExhausted(const std::string &pool) : std::runtime_error(pool + " budget exhausted") {
  }
};

namespace detail {
// Host access must be serialized. An allocation is reusable only after every
// lease is released. Submissions retain leases until native completion.
template <class T>
class ResourcePool {
 public:
  ResourcePool(size_t budget, const char *name) : budget_(budget), name_(name) {
    if (!budget)
      throw std::invalid_argument("resource pool budget must be positive");
  }

  ResourcePool(const ResourcePool &) = delete;
  ResourcePool &operator=(const ResourcePool &) = delete;

  template <class Matches, class Factory>
  std::shared_ptr<T> Acquire(size_t cost, Matches matches, Factory create) {
    if (!cost || cost > budget_)
      throw ResourcePoolExhausted(name_);
    for (auto &entry : entries_)
      if (entry.value.use_count() == 1 && matches(*entry.value))
        return entry.value;
    // Evict idle cached allocations to make room. In-flight leases never move.
    for (auto it = entries_.begin(); used_ > budget_ - cost && it != entries_.end();) {
      if (it->value.use_count() == 1) {
        used_ -= it->cost;
        it = entries_.erase(it);
      } else {
        ++it;
      }
    }
    if (used_ > budget_ - cost)
      throw ResourcePoolExhausted(name_);
    auto value = create();
    entries_.push_back({value, cost});
    used_ += cost;
    return value;
  }

  void Clear() {
    entries_.clear();
    used_ = 0;
  }

 private:
  struct Entry {
    std::shared_ptr<T> value;
    size_t cost;
  };

  size_t budget_, used_{0};
  const char *name_;
  std::vector<Entry> entries_;
};
}  // namespace detail
}  // namespace grassland::graphics
