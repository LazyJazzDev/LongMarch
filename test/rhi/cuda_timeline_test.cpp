#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cmath>
#include <thread>

#include "../../demo/rhi_hello/modules/nbody/cuda_simulation.h"
#include "grassland/rhi/rhi.h"

#if defined(LONGMARCH_CUDA_ENABLED)
cudaError_t TransformCUDABuffer(void *, uint32_t, cudaStream_t);
#endif

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
  ObjPtr<ICUDATimeline> timeline = device->CreateCUDATimeline();
  ASSERT_TRUE(timeline);
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
        waited = timeline->Wait(consumer.stream, {submitted->GetEvent()});
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
  ASSERT_TRUE(timeline->Wait(nullptr, {submitted->GetEvent()}));
  ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
  ASSERT_TRUE(submitted->WaitHost());
}

#if defined(LONGMARCH_CUDA_ENABLED)
TEST_P(RhiCUDATimeline, NBodyCUDAAgreesWithCPUReference) {
  ObjPtr<ICore> core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  if (device->GetCUDADeviceIndex() < 0)
    GTEST_SKIP();
  constexpr uint32_t count = 257;
  constexpr float dt = 0.01f, gravity = 0.4f;
  using Vector = std::array<float, 3>;
  std::vector<Vector> positions(count), velocities(count), expected_positions(count), expected_velocities(count);
  for (uint32_t i = 0; i < count; ++i) {
    positions[i] = {float(i % 7) * 0.3f, float(i % 17) * 0.2f, float(i / 7) * 0.1f};
    velocities[i] = {0.01f, -0.02f, 0.03f};
  }
  for (uint32_t i = 0; i < count; ++i) {
    Vector acceleration{};
    for (uint32_t j = 0; j < count; ++j) {
      Vector difference{};
      float squared = 0.00125f * 0.00125f;
      for (unsigned axis = 0; axis < 3; ++axis) {
        difference[axis] = positions[i][axis] - positions[j][axis];
        squared += difference[axis] * difference[axis];
      }
      const float inverse = 1.0f / std::sqrt(squared);
      for (unsigned axis = 0; axis < 3; ++axis)
        acceleration[axis] += difference[axis] * (inverse * inverse * inverse * (-dt * gravity));
    }
    for (unsigned axis = 0; axis < 3; ++axis) {
      expected_velocities[i][axis] = velocities[i][axis] + acceleration[axis];
      expected_positions[i][axis] = positions[i][axis] + expected_velocities[i][axis] * dt;
    }
  }
  constexpr uint64_t bytes = count * sizeof(Vector);
  ObjPtr<ICUDABuffer> p = device->CreateCUDABuffer({bytes});
  ObjPtr<ICUDABuffer> v = device->CreateCUDABuffer({bytes});
  ObjPtr<ICUDABuffer> next = device->CreateCUDABuffer({bytes});
  ASSERT_TRUE(p && v && next);
  NBodyCUDA cuda(device.Get());
  ASSERT_TRUE(cuda);
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_MAIN);
  ObjPtr<ICommandContext> upload = queue->CreateCommandContext();
  ASSERT_TRUE(upload->CmdUploadBuffer(p.Get(), 0, positions.data(), bytes));
  ASSERT_TRUE(upload->CmdUploadBuffer(v.Get(), 0, velocities.data(), bytes));
  ObjPtr<ISubmissionToken> uploaded = upload->Submit();
  ASSERT_TRUE(uploaded);
  auto ready = cuda.Step(p->GetDevicePointer(), v->GetDevicePointer(), next->GetDevicePointer(), count, dt, gravity,
                         uploaded->GetEvent());
  ASSERT_NE(ready.timeline, nullptr);
  ObjPtr<ICommandContext> commands = queue->CreateCommandContext();
  ObjPtr<IDownloadToken> downloaded_positions = commands->CmdDownloadBuffer(p.Get(), 0, bytes);
  ObjPtr<IDownloadToken> downloaded_velocities = commands->CmdDownloadBuffer(v.Get(), 0, bytes);
  ASSERT_TRUE(downloaded_positions && downloaded_velocities);
  ObjPtr<ISubmissionToken> done = commands->Submit({{ready}});
  ASSERT_TRUE(done && done->WaitHost());
  ASSERT_TRUE(downloaded_positions->CopyTo(positions.data(), bytes));
  ASSERT_TRUE(downloaded_velocities->CopyTo(velocities.data(), bytes));
  for (uint32_t i = 0; i < count; ++i)
    for (unsigned axis = 0; axis < 3; ++axis) {
      EXPECT_NEAR(positions[i][axis], expected_positions[i][axis], 2e-5f);
      EXPECT_NEAR(velocities[i][axis], expected_velocities[i][axis], 2e-5f);
    }
}

TEST_P(RhiCUDATimeline, GraphicsUploadCUDAKernelGraphicsReadback) {
  ObjPtr<ICore> core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  if (device->GetCUDADeviceIndex() < 0)
    GTEST_SKIP();
  ASSERT_EQ(cudaSetDevice(device->GetCUDADeviceIndex()), cudaSuccess);
  ObjPtr<ICUDATimeline> timeline = device->CreateCUDATimeline();
  ASSERT_TRUE(timeline);
  constexpr uint32_t count = 16387;
  ObjPtr<ICUDABuffer> buffer = device->CreateCUDABuffer({count * sizeof(uint32_t)});
  ASSERT_TRUE(buffer);
  auto pointer = buffer->GetDevicePointer();
  CUDAStream stream;
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream.stream, cudaStreamNonBlocking), cudaSuccess);
  std::vector<uint32_t> expected(count), actual(count);
  for (uint32_t round = 0; round < 8; ++round) {
    for (uint32_t i = 0; i < count; ++i)
      expected[i] = i + round * count;
    ObjPtr<IQueue> queue = device->GetQueue(round % 2 ? QUEUE_TYPE_ASSIST : QUEUE_TYPE_MAIN);
    ObjPtr<ICommandContext> upload = queue->CreateCommandContext();
    ASSERT_TRUE(upload->CmdUploadBuffer(buffer.Get(), 0, expected.data(), count * sizeof(uint32_t)));
    ObjPtr<ISubmissionToken> uploaded = upload->Submit();
    ASSERT_TRUE(uploaded);
    ASSERT_TRUE(timeline->Wait(stream.stream, {uploaded->GetEvent()}));
    ASSERT_EQ(TransformCUDABuffer(pointer, count, stream.stream), cudaSuccess);
    auto ready = timeline->Signal(stream.stream, round + 1);
    ASSERT_NE(ready.timeline, nullptr);
    ObjPtr<ICommandContext> readback = queue->CreateCommandContext();
    ObjPtr<IDownloadToken> downloaded = readback->CmdDownloadBuffer(buffer.Get(), 0, count * sizeof(uint32_t));
    ASSERT_TRUE(downloaded);
    ObjPtr<ISubmissionToken> done = readback->Submit({{ready}});
    ASSERT_TRUE(done);
    ASSERT_TRUE(downloaded->WaitHost());
    ASSERT_TRUE(downloaded->CopyTo(actual.data(), count * sizeof(uint32_t)));
    for (auto &value : expected)
      value = value * 3 + 7;
    EXPECT_EQ(actual, expected);
    EXPECT_EQ(buffer->GetDevicePointer(), pointer);
  }
}
#endif

TEST_P(RhiCUDATimeline, WaitsForAllEventsAndCoalescesDuplicateTimelines) {
  ObjPtr<ICore> core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  if (device->GetCUDADeviceIndex() < 0)
    GTEST_SKIP();
  ASSERT_EQ(cudaSetDevice(device->GetCUDADeviceIndex()), cudaSuccess);
  ObjPtr<ICUDATimeline> first = device->CreateCUDATimeline();
  ObjPtr<ICUDATimeline> second = device->CreateCUDATimeline();
  ObjPtr<ICUDATimeline> joined = device->CreateCUDATimeline();
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  ASSERT_TRUE(joined);
  EXPECT_EQ(QueryInterfacePtr<ITimeline>(first.Get()).Get(), first.Get());
  EXPECT_EQ(QueryInterfacePtr<IObject>(first.Get()).Get(),
            QueryInterfacePtr<IObject>(QueryInterfacePtr<ITimeline>(first.Get()).Get()).Get());
  CUDAStream a, b, consumer;
  ASSERT_EQ(cudaStreamCreateWithFlags(&a.stream, cudaStreamNonBlocking), cudaSuccess);
  ASSERT_EQ(cudaStreamCreateWithFlags(&b.stream, cudaStreamNonBlocking), cudaSuccess);
  ASSERT_EQ(cudaStreamCreateWithFlags(&consumer.stream, cudaStreamNonBlocking), cudaSuccess);
  EXPECT_TRUE(joined->Wait(consumer.stream, {}));
  EXPECT_FALSE(joined->Wait(consumer.stream, {{nullptr, 1}}));
  EXPECT_EQ(first->Signal(a.stream, 0).timeline, nullptr);
  Gate gate;
  ASSERT_EQ(cudaLaunchHostFunc(b.stream, Gate::Wait, &gate), cudaSuccess);
  auto event_a = first->Signal(a.stream, 2);
  auto event_b = second->Signal(b.stream, 1);
  ObjPtr<IQueue> main_queue = device->GetQueue(QUEUE_TYPE_MAIN);
  ObjPtr<IQueue> assist_queue = device->GetQueue(QUEUE_TYPE_ASSIST);
  ObjPtr<ICommandContext> main_commands = main_queue->CreateCommandContext();
  ObjPtr<ICommandContext> assist_commands = assist_queue->CreateCommandContext();
  ObjPtr<ISubmissionToken> main_done, assist_done;
  if (event_a.timeline && main_commands)
    main_done = main_commands->Submit({{event_a}});
  if (event_b.timeline && assist_commands)
    assist_done = assist_commands->Submit({{event_b}});
  bool waited =
      main_done && assist_done &&
      joined->Wait(consumer.stream, {{first.Get(), 1}, event_a, main_done->GetEvent(), assist_done->GetEvent()});
  auto ready = waited ? joined->Signal(consumer.stream, 1) : Event{};
  EXPECT_EQ(cudaStreamSynchronize(a.stream), cudaSuccess);
  EXPECT_EQ(cudaStreamQuery(consumer.stream), cudaErrorNotReady);
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_MAIN);
  ObjPtr<ICommandContext> commands = queue->CreateCommandContext();
  ObjPtr<ISubmissionToken> submitted;
  if (ready.timeline && commands)
    submitted = commands->Submit({{ready}});
  if (submitted)
    EXPECT_EQ(submitted->GetStatus(), SUBMISSION_STATUS_PENDING);
  gate.open = true;
  EXPECT_EQ(cudaStreamSynchronize(b.stream), cudaSuccess);
  EXPECT_EQ(cudaStreamSynchronize(consumer.stream), cudaSuccess);
  ASSERT_TRUE(waited);
  ASSERT_TRUE(submitted);
  EXPECT_TRUE(submitted->WaitHost());
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
#if defined(LONGMARCH_VULKAN_ENABLED)
    BACKEND_API_VULKAN,
#endif
};
INSTANTIATE_TEST_SUITE_P(Backends, RhiCUDATimeline, testing::ValuesIn(backends));
}  // namespace
}  // namespace grassland::rhi
