#pragma once
#include "cstdint"
#include "functional"
#include "map"
#include "set"

namespace grassland {
template <class Func>
class EventManager {
 public:
  uint32_t RegisterCallback(const std::function<Func> &callback,
                            int priority = 100) {
    callbacks_[callback_counter_] = std::make_pair(callback, priority);
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
    for (auto &priority : priority_map_) {
      for (auto &id : priority.second) {
        callbacks_[id].first(std::forward<Args>(args)...);
      }
    }
  }

 private:
  uint32_t callback_counter_{0};
  std::map<uint32_t, std::pair<std::function<Func>, int>> callbacks_{};
  std::map<int, std::set<uint32_t>, std::greater<>> priority_map_{};
};
}  // namespace grassland
