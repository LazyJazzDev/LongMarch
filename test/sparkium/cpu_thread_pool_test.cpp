#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <set>

#include "sparkium/backend/cpu/native_cpu_thread_pool.h"

using sparkium::backend::NativeCpuThreadPool;

TEST(NativeCpuThreadPoolTest, CoversTailExactlyOnceAndAcceptsEmptyDispatch) {
  NativeCpuThreadPool pool(4);
  for (unsigned count : {0, 1, 16, 17, 97, 257}) {
    std::array<std::atomic<unsigned>, 257> hits{};
    for (auto &hit : hits)
      hit.store(0);
    pool.Run(count, [&](uint64_t begin, uint64_t end) {
      for (auto i = begin; i < end; ++i)
        ++hits.at(i);
    });
    for (unsigned i = 0; i < hits.size(); ++i)
      EXPECT_EQ(hits[i].load(), unsigned(i < count));
  }
}

TEST(NativeCpuThreadPoolTest, ReusesTheSameWorkersBetweenSubmissions) {
  NativeCpuThreadPool pool(4);
  std::set<std::thread::id> previous;
  for (int repeat = 0; repeat < 3; ++repeat) {
    std::mutex mutex;
    std::condition_variable ready;
    std::set<std::thread::id> current;
    pool.Run(32, [&](uint64_t, uint64_t) {
      std::unique_lock<std::mutex> lock(mutex);
      current.insert(std::this_thread::get_id());
      ready.notify_all();
      if (!ready.wait_for(lock, std::chrono::seconds(5), [&] { return current.size() == 4; }))
        throw std::runtime_error("workers did not rendezvous");
    });
    ASSERT_EQ(current.size(), 4);
    if (repeat)
      EXPECT_EQ(current, previous);
    previous = current;
  }
}

TEST(NativeCpuThreadPoolTest, PropagatesFailureAndRemainsReusable) {
  NativeCpuThreadPool pool(4);
  EXPECT_THROW(pool.Run(100, [](uint64_t, uint64_t) { throw std::runtime_error("task failure"); }), std::runtime_error);
  std::atomic<unsigned> total{0};
  EXPECT_NO_THROW(pool.Run(103, [&](uint64_t begin, uint64_t end) { total += unsigned(end - begin); }));
  EXPECT_EQ(total.load(), 103);
}

TEST(NativeCpuThreadPoolTest, SerializesConcurrentSubmittersWithoutLosingWork) {
  NativeCpuThreadPool pool(4);
  std::array<std::atomic<unsigned>, 2> totals{};
  for (auto &total : totals)
    total.store(0);
  auto submit = [&](int index) {
    for (int i = 0; i < 20; ++i)
      pool.Run(101, [&](uint64_t begin, uint64_t end) { totals[index] += unsigned(end - begin); });
  };

  std::thread a(submit, 0), b(submit, 1);
  a.join();
  b.join();
  EXPECT_EQ(totals[0].load(), 2020);
  EXPECT_EQ(totals[1].load(), 2020);
}

TEST(NativeCpuThreadPoolTest, SingleThreadRunsOnTheCaller) {
  EXPECT_THROW(NativeCpuThreadPool(0), std::invalid_argument);
  NativeCpuThreadPool pool(1);
  const auto caller = std::this_thread::get_id();
  pool.Run(100, [&](uint64_t begin, uint64_t end) {
    EXPECT_EQ(std::this_thread::get_id(), caller);
    EXPECT_EQ(begin, 0);
    EXPECT_EQ(end, 100);
  });
}

TEST(NativeCpuThreadPoolTest, ConfigurableGrainPreservesExactCoverage) {
  EXPECT_THROW(NativeCpuThreadPool(4, 0), std::invalid_argument);
  for (uint64_t grain : {1u, 7u, 32u, 1000u}) {
    NativeCpuThreadPool pool(4, grain);
    std::array<std::atomic<unsigned>, 97> hits{};
    for (auto &hit : hits)
      hit = 0;
    pool.Run(hits.size(), [&](uint64_t begin, uint64_t end) {
      EXPECT_LE(end - begin, grain);
      for (auto i = begin; i < end; ++i)
        ++hits.at(i);
    });
    for (auto &hit : hits)
      EXPECT_EQ(hit.load(), 1u);
  }
}
