#include <gtest/gtest.h>

#include <atomic>
#include <thread>

#include "grassland/rhi/rhi.h"

namespace grassland::rhi {
namespace {
struct CUDAStream {
  cudaStream_t stream{};

  ~CUDAStream() {
    if (stream) {
      cudaStreamSynchronize(stream);
      cudaStreamDestroy(stream);
    }
  }
};

struct Gate {
  std::atomic<bool> open{};

  ~Gate() {
    open = true;
  }

  static void CUDART_CB Wait(void *data) {
    auto gate = static_cast<Gate *>(data);
    while (!gate->open.load())
      std::this_thread::yield();
  }
};

class RhiCUDATimeline : public testing::TestWithParam<BackendAPI> {};

TEST_P(RhiCUDATimeline, BothQueuesWaitForCUDAAndCUDAWaitsForQueueEvents) {
  ObjPtr<ICore> core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  if (device->GetCUDADeviceIndex() < 0)
    GTEST_SKIP();
  ASSERT_EQ(cudaSetDevice(device->GetCUDADeviceIndex()), cudaSuccess);
  auto timeline = device->CreateCUDATimeline();
  ASSERT_NE(timeline, nullptr);
  CUDAStream producer, consumer;
  ASSERT_EQ(cudaStreamCreateWithFlags(&producer.stream, cudaStreamNonBlocking), cudaSuccess);
  ASSERT_EQ(cudaStreamCreateWithFlags(&consumer.stream, cudaStreamNonBlocking), cudaSuccess);
  for (auto type : {QUEUE_TYPE_MAIN, QUEUE_TYPE_ASSIST}) {
    ObjPtr<IQueue> queue = device->GetQueue(type);
    ObjPtr<ICommandContext> commands = queue->CreateCommandContext();
    ASSERT_TRUE(commands);
    Gate gate;
    ASSERT_EQ(cudaLaunchHostFunc(producer.stream, Gate::Wait, &gate), cudaSuccess);
    auto ready = timeline->Signal(producer.stream, uint64_t(type) + 1);
    EXPECT_NE(ready.timeline, nullptr);
    ObjPtr<ISubmissionToken> submitted;
    bool waited = false;
    cudaError_t query = cudaSuccess;
    if (ready.timeline) {
      submitted = commands->Submit({{ready}});
      if (submitted) {
        waited = timeline->Wait(consumer.stream, submitted->GetEvent());
        query = cudaStreamQuery(consumer.stream);
        EXPECT_EQ(submitted->GetStatus(), SUBMISSION_STATUS_PENDING);
      }
    }
    gate.open = true;
    ASSERT_EQ(cudaStreamSynchronize(producer.stream), cudaSuccess);
    ASSERT_TRUE(submitted);
    ASSERT_TRUE(waited);
    EXPECT_EQ(query, cudaErrorNotReady);
    ASSERT_EQ(cudaStreamSynchronize(consumer.stream), cudaSuccess);
    ASSERT_TRUE(submitted->WaitHost());
    uint64_t completed = 0;
    ASSERT_TRUE(timeline->GetCompletedValue(completed));
    EXPECT_GE(completed, ready.value);
  }
  // Null is a real CUDA default stream, not "no synchronization".
  auto ready = timeline->Signal(nullptr, 3);
  ASSERT_NE(ready.timeline, nullptr);
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_MAIN);
  ObjPtr<ICommandContext> commands = queue->CreateCommandContext();
  ObjPtr<ISubmissionToken> submitted = commands->Submit({{ready}});
  ASSERT_TRUE(submitted);
  ASSERT_TRUE(timeline->Wait(nullptr, submitted->GetEvent()));
  ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
  ASSERT_TRUE(submitted->WaitHost());
}

TEST_P(RhiCUDATimeline, NonCUDADevicesRejectTimelineCreation) {
  ObjPtr<ICore> core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  const auto adapters = core->GetPhysicalDevices();
  unsigned tested = 0;
  for (size_t i = 0; i < adapters.size(); ++i) {
    if (adapters[i].cuda_device >= 0)
      continue;
    ObjPtr<IDevice> device = core->CreateDevice(int32_t(i));
    if (!device)
      continue;
    ++tested;
    EXPECT_EQ(device->CreateCUDATimeline(), nullptr);
  }
  if (!tested)
    GTEST_SKIP();
}

const BackendAPI backends[]{
#if defined(LONGMARCH_D3D12_ENABLED)
    BACKEND_API_D3D12,
#endif
    BACKEND_API_VULKAN};
INSTANTIATE_TEST_SUITE_P(Backends, RhiCUDATimeline, testing::ValuesIn(backends));
}  // namespace
}  // namespace grassland::rhi
