#pragma once
#include <memory>
#include <vector>

#include "cstdint"
#include "functional"
#include "map"
#include "set"

namespace grassland {
template <class Func>
class EventManager {
 public:
  EventManager() = default;

  EventManager(const EventManager &other)
      : callback_counter_(other.callback_counter_),
        priority_map_(other.priority_map_) {
    for (const auto &entry : other.callbacks_)
      callbacks_.emplace(
          entry.first, std::make_pair(std::make_shared<std::function<Func>>(*entry.second.first), entry.second.second));
  }

  EventManager &operator=(const EventManager &other) {
    if (this != &other) {
      EventManager copy(other);
      *this = std::move(copy);
    }
    return *this;
  }

  EventManager(EventManager &&) = default;
  EventManager &operator=(EventManager &&) = default;

  uint32_t RegisterCallback(const std::function<Func> &callback, int priority = 100) {
    callbacks_[callback_counter_] = std::make_pair(std::make_shared<std::function<Func>>(callback), priority);
    priority_map_[priority].insert(callback_counter_);
    return callback_counter_++;
  }

  void UnregisterCallback(uint32_t id) {
    auto it = callbacks_.find(id);
    if (it != callbacks_.end()) {
      priority_map_[it->second.second].erase(id);
      callbacks_.erase(it);
    }
  }

  template <class... Args>
  void InvokeCallbacks(Args &&...args) {
    // Snapshot IDs so callbacks may unregister themselves or other callbacks.
    // New registrations participate in the next invocation, not this one.
    std::vector<uint32_t> pending;
    for (const auto &priority : priority_map_)
      pending.insert(pending.end(), priority.second.begin(), priority.second.end());
    for (auto id : pending) {
      const auto it = callbacks_.find(id);
      if (it == callbacks_.end())
        continue;
      // Keep the callable alive if it unregisters itself while executing.
      auto callback = it->second.first;
      (*callback)(std::forward<Args>(args)...);
    }
  }

 private:
  uint32_t callback_counter_{0};
  std::map<uint32_t, std::pair<std::shared_ptr<std::function<Func>>, int>> callbacks_{};
  std::map<int, std::set<uint32_t>, std::greater<>> priority_map_{};
};
}  // namespace grassland
