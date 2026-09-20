#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace sparkium::backend {
// Synchronous range execution. Workers persist across shaders and frames;
// concurrent submitters are serialized and the submitting thread also works.
class NativeCpuThreadPool {
 public:
  explicit NativeCpuThreadPool(unsigned thread_count, uint64_t grain = 8);
  ~NativeCpuThreadPool();
  NativeCpuThreadPool(const NativeCpuThreadPool &) = delete;
  NativeCpuThreadPool &operator=(const NativeCpuThreadPool &) = delete;
  void Run(uint64_t count, const std::function<void(uint64_t, uint64_t)> &function);
  static NativeCpuThreadPool &Shared();

 private:
  void Worker();
  void Execute();
  std::mutex submit_mutex_, state_mutex_;
  std::condition_variable ready_, finished_;
  std::vector<std::thread> workers_;
  std::function<void(uint64_t, uint64_t)> function_;
  std::atomic<uint64_t> next_{0};
  std::atomic<bool> cancelled_{false};
  const uint64_t grain_;
  uint64_t count_{0}, generation_{0};
  size_t remaining_{0};
  bool stopping_{false};
  std::exception_ptr error_;
};
}  // namespace sparkium::backend
