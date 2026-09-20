#include "sparkium/backend/cpu/native_cpu_thread_pool.h"

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace sparkium::backend {
namespace {
unsigned ThreadCount() {
  unsigned count = std::max(1u, std::thread::hardware_concurrency());
  const char *value = std::getenv("SPARKIUM_CPU_THREADS");
  if (!value)
    value = std::getenv("OMP_NUM_THREADS");  // Compatibility with earlier CPU releases.
  if (value) {
    unsigned requested = 0;
    auto result = std::from_chars(value, value + std::strlen(value), requested);
    if (result.ec != std::errc{} || *result.ptr != '\0' || !requested)
      throw std::runtime_error("CPU thread count must be a positive integer");
    count = std::min(count, requested);
  }
  return count;
}

uint64_t Grain() {
  const char *value = std::getenv("SPARKIUM_CPU_GRAIN");
  if (!value)
    return 8;
  uint64_t grain = 0;
  auto result = std::from_chars(value, value + std::strlen(value), grain);
  if (result.ec != std::errc{} || *result.ptr != '\0' || !grain)
    throw std::runtime_error("CPU grain must be a positive integer");
  return grain;
}
}  // namespace

NativeCpuThreadPool &NativeCpuThreadPool::Shared() {
  static NativeCpuThreadPool pool(ThreadCount(), Grain());
  return pool;
}

NativeCpuThreadPool::NativeCpuThreadPool(unsigned thread_count, uint64_t grain) : grain_(grain) {
  if (!grain)
    throw std::invalid_argument("CPU grain must be positive");
  if (!thread_count)
    throw std::invalid_argument("CPU thread count must be positive");
  try {
    for (unsigned i = 1; i < thread_count; ++i)
      workers_.emplace_back([this] { Worker(); });
  } catch (...) {
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      stopping_ = true;
    }
    ready_.notify_all();
    for (auto &worker : workers_)
      worker.join();
    throw;
  }
}

NativeCpuThreadPool::~NativeCpuThreadPool() {
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    stopping_ = true;
  }
  ready_.notify_all();
  for (auto &worker : workers_)
    worker.join();
}

void NativeCpuThreadPool::Execute() {
  try {
    while (!cancelled_.load(std::memory_order_relaxed)) {
      uint64_t begin = next_.load(std::memory_order_relaxed);
      if (begin >= count_)
        break;
      const uint64_t end = begin + std::min(grain_, count_ - begin);
      if (!next_.compare_exchange_weak(begin, end, std::memory_order_relaxed))
        continue;
      function_(begin, end);
    }
  } catch (...) {
    cancelled_.store(true, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (!error_)
      error_ = std::current_exception();
  }
}

void NativeCpuThreadPool::Worker() {
  uint64_t seen = 0;
  std::unique_lock<std::mutex> lock(state_mutex_);
  for (;;) {
    ready_.wait(lock, [&] { return stopping_ || generation_ != seen; });
    if (stopping_)
      return;
    seen = generation_;
    lock.unlock();
    Execute();
    lock.lock();
    if (--remaining_ == 0)
      finished_.notify_one();
  }
}

void NativeCpuThreadPool::Run(uint64_t count, const std::function<void(uint64_t, uint64_t)> &function) {
  std::lock_guard<std::mutex> submit(submit_mutex_);
  if (!count)
    return;
  if (workers_.empty() || count <= 16) {
    function(0, count);
    return;
  }
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    function_ = function;
    count_ = count;
    next_.store(0, std::memory_order_relaxed);
    cancelled_.store(false, std::memory_order_relaxed);
    error_ = nullptr;
    remaining_ = workers_.size();
    ++generation_;
  }
  ready_.notify_all();
  Execute();
  std::unique_lock<std::mutex> lock(state_mutex_);
  finished_.wait(lock, [&] { return remaining_ == 0; });
  function_ = {};
  if (error_)
    std::rethrow_exception(error_);
}
}  // namespace sparkium::backend
