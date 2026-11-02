#include <gtest/gtest.h>

#include <numeric>

#include "grassland/rhi/rhi.h"
#include "grassland/rhi/rhix/rhix.h"

#if defined(LONGMARCH_CUDA_RUNTIME)
#if defined(LONGMARCH_CUDA_ENABLED)
cudaError_t AddCUDAInteropValues(void *, uint32_t, uint32_t, cudaStream_t);
#endif
namespace grassland::rhi {
namespace {
class RhiCUDAInterop : public testing::TestWithParam<BackendAPI> {};

struct Streams {
  cudaStream_t a{}, b{};

  ~Streams() {
    if (a) {
      cudaStreamSynchronize(a);
      cudaStreamDestroy(a);
    }
    if (b) {
      cudaStreamSynchronize(b);
      cudaStreamDestroy(b);
    }
  }
};

struct RestoreDevice {
  int previous{-1};

  RestoreDevice() {
    cudaGetDevice(&previous);
  }

  ~RestoreDevice() {
    if (previous >= 0)
      cudaSetDevice(previous);
  }
};

TEST_P(RhiCUDAInterop, MatchesDevicesAndRejectsUnsupportedAdapters) {
  ObjPtr<ICore> core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  for (size_t i = 0; i < core->GetPhysicalDevices().size(); ++i) {
    const auto properties = core->GetPhysicalDevices()[i];
    ObjPtr<IDevice> device = core->CreateDevice(static_cast<int32_t>(i));
    ASSERT_TRUE(device);
    EXPECT_EQ(device->GetProperties().cuda_device, properties.cuda_device);
    EXPECT_EQ(device->GetProperties().name, properties.name);
    if (properties.cuda_device < 0) {
      EXPECT_EQ(device->CreateCUDABuffer({4096}), nullptr);
      EXPECT_EQ(device->CreateCUDAInteropTimeline(), nullptr);
    } else {
      int count{};
      ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
      EXPECT_LT(properties.cuda_device, count);
    }
  }
}

TEST_P(RhiCUDAInterop, MapsOnceForTheBufferLifetime) {
  ObjPtr<ICore> core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  if (device->GetProperties().cuda_device < 0)
    GTEST_SKIP() << "No matching CUDA device";
  EXPECT_EQ(device->CreateCUDABuffer({0}), nullptr);
  ObjPtr<ICUDABuffer> buffer = device->CreateCUDABuffer({65549});
  ASSERT_TRUE(buffer);
  EXPECT_EQ(buffer->GetDesc().size, 65549);
  auto pointer = buffer->GetDevicePointer();
  ASSERT_NE(pointer, nullptr);
  for (int i = 0; i < 100; ++i)
    EXPECT_EQ(pointer, buffer->GetDevicePointer());
  cudaPointerAttributes attributes{};
  ASSERT_EQ(cudaPointerGetAttributes(&attributes, pointer), cudaSuccess);
  EXPECT_EQ(attributes.device, device->GetProperties().cuda_device);
  EXPECT_EQ(attributes.type, cudaMemoryTypeDevice);
  ObjPtr<IBuffer> ordinary = buffer;
  EXPECT_EQ(ordinary->GetDesc().size, 65549);
}

TEST_P(RhiCUDAInterop, RejectsInvalidSubmissionDependencies) {
  ObjPtr<ICore> core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  if (device->GetProperties().cuda_device < 0)
    GTEST_SKIP();
  ObjPtr<ICUDAInteropTimeline> timeline = device->CreateCUDAInteropTimeline();
  ASSERT_TRUE(timeline);
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  auto reject = [&](const SubmitInfo &info) {
    ObjPtr<ICommandContext> commands = device->CreateCommandContext();
    ObjPtr<ISubmissionToken> token = queue->Submit(commands.Get(), info);
    EXPECT_FALSE(token);
  };
  SubmitInfo info;
  info.cuda_waits = {{nullptr, 1}};
  reject(info);
  info.cuda_waits = {{timeline.Get(), 1}, {timeline.Get(), 2}};
  reject(info);
  info.cuda_waits.clear();
  info.cuda_signals = {{timeline.Get(), 0}};
  reject(info);
  info.cuda_waits = {{timeline.Get(), 1}};
  info.cuda_signals = {{timeline.Get(), 1}};
  reject(info);
  ObjPtr<IDevice> other = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(other);
  ObjPtr<ICUDAInteropTimeline> foreign = other->CreateCUDAInteropTimeline();
  ASSERT_TRUE(foreign);
  info.cuda_waits = {{foreign.Get(), 1}};
  info.cuda_signals.clear();
  reject(info);
}

TEST_P(RhiCUDAInterop, RHIXWaitsAndSignalsMultipleTimelines) {
  RestoreDevice restore;
  grassland::rhix::Core core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  auto device = core.CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  if (device.GetProperties().cuda_device < 0)
    GTEST_SKIP();
  ASSERT_EQ(cudaSetDevice(device.GetProperties().cuda_device), cudaSuccess);
  auto buffer = device.CreateCUDABuffer({4096});
  ASSERT_TRUE(buffer);
  ASSERT_NE(buffer.GetDevicePointer(), nullptr);
  EXPECT_EQ(buffer.GetDesc().size, 4096);
  auto a = device.CreateCUDAInteropTimeline();
  auto b = device.CreateCUDAInteropTimeline();
  ASSERT_TRUE(a);
  ASSERT_TRUE(b);
  ASSERT_TRUE(a.SignalCUDA(nullptr, 1));
  ASSERT_TRUE(b.SignalCUDA(nullptr, 1));
  auto queue = device.GetQueue(QUEUE_TYPE_GRAPHICS);
  grassland::rhix::SubmitInfo info;
  info.cuda_waits = {{a, 1}, {b, 1}};
  info.cuda_signals = {{a, 2}, {b, 2}};
  auto token = queue.Submit(device.CreateCommandContext(), info);
  ASSERT_TRUE(token);
  ASSERT_TRUE(a.WaitCUDA(nullptr, 2));
  ASSERT_TRUE(b.WaitCUDA(nullptr, 2));
  ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
  ASSERT_TRUE(token.WaitHost());
}

#if defined(LONGMARCH_CUDA_ENABLED)
TEST_P(RhiCUDAInterop, TwoStreamsRoundTripWithoutHostWaitBetweenSubmissions) {
  RestoreDevice restore;
  ObjPtr<ICore> core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  const int ordinal = device->GetProperties().cuda_device;
  if (ordinal < 0)
    GTEST_SKIP() << "No matching CUDA device";
  ASSERT_EQ(cudaSetDevice(ordinal), cudaSuccess);
  ObjPtr<ICUDABuffer> a = device->CreateCUDABuffer({4096});
  ObjPtr<ICUDABuffer> b = device->CreateCUDABuffer({4096});
  ObjPtr<ICUDAInteropTimeline> graphics = device->CreateCUDAInteropTimeline();
  ObjPtr<ICUDAInteropTimeline> ready_a = device->CreateCUDAInteropTimeline();
  ObjPtr<ICUDAInteropTimeline> ready_b = device->CreateCUDAInteropTimeline();
  ASSERT_TRUE(a);
  ASSERT_TRUE(b);
  ASSERT_TRUE(graphics);
  ASSERT_TRUE(ready_a);
  ASSERT_TRUE(ready_b);
  Streams streams;
  ASSERT_EQ(cudaStreamCreateWithFlags(&streams.a, cudaStreamNonBlocking), cudaSuccess);
  ASSERT_EQ(cudaStreamCreateWithFlags(&streams.b, cudaStreamNonBlocking), cudaSuccess);
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  std::vector<uint32_t> input(1024);
  std::iota(input.begin(), input.end(), 0);
  ObjPtr<ICommandContext> upload = device->CreateCommandContext();
  ASSERT_TRUE(upload);
  ASSERT_TRUE(upload->CmdUploadBuffer(a.Get(), 0, input.data(), 4096));
  ASSERT_TRUE(upload->CmdUploadBuffer(b.Get(), 0, input.data(), 4096));
  ASSERT_TRUE(upload->CmdReleaseCUDABuffer(a.Get()));
  ASSERT_TRUE(upload->CmdReleaseCUDABuffer(b.Get()));
  SubmitInfo first;
  first.cuda_signals = {{graphics.Get(), 1}};
  ObjPtr<ISubmissionToken> initial = queue->Submit(upload.Get(), first);
  ASSERT_TRUE(initial);
  std::vector<ObjPtr<IDownloadToken>> results_a, results_b;
  ObjPtr<ISubmissionToken> final;
  for (uint64_t round = 1; round <= 4; ++round) {
    ASSERT_TRUE(graphics->WaitCUDA(streams.a, round));
    ASSERT_TRUE(graphics->WaitCUDA(streams.b, round));
    ASSERT_EQ(AddCUDAInteropValues(a->GetDevicePointer(), 1024, 7, streams.a), cudaSuccess);
    ASSERT_EQ(AddCUDAInteropValues(b->GetDevicePointer(), 1024, 11, streams.b), cudaSuccess);
    ASSERT_TRUE(ready_a->SignalCUDA(streams.a, round));
    ASSERT_TRUE(ready_b->SignalCUDA(streams.b, round));
    ObjPtr<ICommandContext> commands = device->CreateCommandContext();
    ASSERT_TRUE(commands);
    ASSERT_TRUE(commands->CmdAcquireCUDABuffer(a.Get()));
    ASSERT_TRUE(commands->CmdAcquireCUDABuffer(b.Get()));
    results_a.emplace_back(commands->CmdDownloadBuffer(a.Get(), 0, 4096));
    results_b.emplace_back(commands->CmdDownloadBuffer(b.Get(), 0, 4096));
    ASSERT_TRUE(results_a.back());
    ASSERT_TRUE(results_b.back());
    if (round < 4) {
      ASSERT_TRUE(commands->CmdReleaseCUDABuffer(a.Get()));
      ASSERT_TRUE(commands->CmdReleaseCUDABuffer(b.Get()));
    }
    SubmitInfo info;
    info.cuda_waits = {{ready_a.Get(), round}, {ready_b.Get(), round}};
    info.cuda_signals = {{graphics.Get(), round + 1}};
    final = queue->Submit(commands.Get(), info);
    ASSERT_TRUE(final);
  }
  ASSERT_TRUE(final->WaitHost());
  // Default stream (0) is a valid wait/signal destination as well.
  ASSERT_TRUE(graphics->WaitCUDA(nullptr, 5));
  ASSERT_TRUE(ready_a->SignalCUDA(nullptr, 5));
  ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
  for (uint32_t round = 1; round <= 4; ++round) {
    std::vector<uint32_t> output_a(1024), output_b(1024);
    ASSERT_TRUE(results_a[round - 1]->CopyTo(output_a.data(), 4096));
    ASSERT_TRUE(results_b[round - 1]->CopyTo(output_b.data(), 4096));
    for (uint32_t i = 0; i < 1024; ++i) {
      ASSERT_EQ(output_a[i], i + 7 * round);
      ASSERT_EQ(output_b[i], i + 11 * round);
    }
  }
}
#endif
std::vector<BackendAPI> CUDABackends() {
  std::vector<BackendAPI> result;
#if defined(LONGMARCH_D3D12_ENABLED)
  result.push_back(BACKEND_API_D3D12);
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
  result.push_back(BACKEND_API_VULKAN);
#endif
  return result;
}

INSTANTIATE_TEST_SUITE_P(Backends, RhiCUDAInterop, testing::ValuesIn(CUDABackends()));
}  // namespace
}  // namespace grassland::rhi
#endif
