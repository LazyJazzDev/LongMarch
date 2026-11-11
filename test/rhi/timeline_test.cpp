#include <gtest/gtest.h>

#include <functional>
#include <future>

#include "grassland/rhi/rhi.h"

#ifdef LONGMARCH_VULKAN_ENABLED
#include "grassland/rhi/backend/vulkan/vulkan_device.h"
#include "grassland/rhi/backend/vulkan/vulkan_queue.h"
#include "grassland/rhi/backend/vulkan/vulkan_timeline.h"
#endif
#ifdef LONGMARCH_D3D12_ENABLED
#include "grassland/rhi/backend/d3d12/d3d12_device.h"
#include "grassland/rhi/backend/d3d12/d3d12_timeline.h"
#endif

#ifdef LONGMARCH_METAL_ENABLED
#include "grassland/rhi/backend/metal/metal_device.h"
#include "grassland/rhi/backend/metal/metal_queue.h"
#include "grassland/rhi/backend/metal/metal_timeline.h"
#endif

namespace grassland::rhi {

namespace {
void CheckIndependentRetirement(BackendAPI api) {
  ObjPtr<ICore> core = CreateCore(api);
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  ObjPtr<IQueue> main = device->GetQueue(QUEUE_TYPE_MAIN);
  ObjPtr<IQueue> assist = device->GetQueue(QUEUE_TYPE_ASSIST);
  ASSERT_TRUE(main);
  ASSERT_TRUE(assist);
#ifdef LONGMARCH_VULKAN_ENABLED
  if (api == BACKEND_API_VULKAN && static_cast<backend::vulkan::IVulkanQueue *>(main.Get())->NativeQueue() ==
                                       static_cast<backend::vulkan::IVulkanQueue *>(assist.Get())->NativeQueue())
    GTEST_SKIP() << "Independent GPU progress requires distinct native queues.";
#endif
  backend::ObjectPtr<backend::Timeline> gate;

  struct GateRelease {
    std::function<void()> open;
    ObjPtr<ISubmissionToken> pending;

    void Open() {
      if (auto call = std::move(open))
        call();
    }

    ~GateRelease() {
      Open();
      if (pending)
        pending->WaitHost();
    }
  } release;
#ifdef LONGMARCH_D3D12_ENABLED
  if (api == BACKEND_API_D3D12) {
    auto timeline = backend::d3d12::ID3D12Timeline::Create(static_cast<backend::d3d12::ID3D12Device *>(device.Get()));
    ASSERT_TRUE(timeline);
    auto fence = timeline->NativeFence();
    release.open = [fence] { EXPECT_EQ(fence->Signal(1), S_OK); };
    gate = std::move(timeline);
  }
#endif
#ifdef LONGMARCH_VULKAN_ENABLED
  if (api == BACKEND_API_VULKAN) {
    auto native = static_cast<backend::vulkan::IVulkanDevice *>(device.Get());
    auto timeline = backend::vulkan::IVulkanTimeline::Create(native);
    ASSERT_TRUE(timeline);
    const auto semaphore = timeline->NativeSemaphore();
    release.open = [native, semaphore] {
      VkSemaphoreSignalInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
      info.semaphore = semaphore;
      info.value = 1;
      EXPECT_EQ(vkSignalSemaphore(native->NativeDevice(), &info), VK_SUCCESS);
    };
    gate = std::move(timeline);
  }
#endif
#ifdef LONGMARCH_METAL_ENABLED
  if (api == BACKEND_API_METAL) {
    auto timeline = backend::metal::IMetalTimeline::Create(static_cast<backend::metal::IMetalDevice *>(device.Get()));
    ASSERT_TRUE(timeline);
    auto native = static_cast<backend::metal::IMetalDevice *>(device.Get());
    auto event = NS::TransferPtr(native->NativeDevice()->newSharedEvent());
    ASSERT_TRUE(event);
    // A CPU-controlled gate must use a SharedEvent. Bridge it to the GPU-only
    // timeline on a separate queue, leaving Main free to make progress.
    auto gate_queue = NS::TransferPtr(native->NativeDevice()->newCommandQueue());
    ASSERT_TRUE(gate_queue);
    auto commands = NS::RetainPtr(gate_queue->commandBuffer());
    ASSERT_TRUE(commands);
    commands->encodeWait(event.get(), 1);
    commands->encodeSignalEvent(timeline->NativeEvent(), 1);
    commands->commit();
    release.open = [event, gate_queue, commands] {
      event->setSignaledValue(1);
      commands->waitUntilCompleted();
    };
    gate = std::move(timeline);
  }
#endif
  ASSERT_TRUE(gate);
  ObjPtr<ICommandContext> blocked = assist->CreateCommandContext();
  ASSERT_TRUE(blocked);
  ObjPtr<ISubmissionToken> assist_done = blocked->Submit({{{gate.Get(), 1}}});
  ASSERT_TRUE(assist_done);
  release.pending = ObjPtr<ISubmissionToken>::Retain(assist_done.Get());
  blocked.Release();
  ObjPtr<ICommandContext> independent = main->CreateCommandContext();
  ASSERT_TRUE(independent);
  ObjPtr<ISubmissionToken> main_done = independent->Submit();
  ASSERT_TRUE(main_done);
  independent.Release();
  auto retired = std::async(std::launch::async, [&] { return main_done->WaitHost(); });
  const auto retired_status = retired.wait_for(std::chrono::seconds(2));
  const auto main_event = main_done->GetEvent();
  const auto assist_event = assist_done->GetEvent();
  auto any = std::async(std::launch::async, [&] {
    return device->WaitEvents({assist_event, {main_event.timeline, main_event.value + 1}, main_event}, false);
  });
  const auto any_status = any.wait_for(std::chrono::seconds(2));
  auto all = std::async(std::launch::async,
                        [&] { return device->WaitEvents({{assist_event.timeline, 0}, assist_event, main_event}); });
  const auto all_status = all.wait_for(std::chrono::milliseconds(20));
  EXPECT_EQ(assist_done->GetStatus(), SUBMISSION_STATUS_PENDING);
  release.Open();
  ObjPtr<ICommandContext> cleanup = main->CreateCommandContext();
  ObjPtr<ISubmissionToken> cleanup_done = cleanup ? cleanup->Submit() : nullptr;
  EXPECT_TRUE(cleanup_done);
  EXPECT_EQ(retired_status, std::future_status::ready);
  EXPECT_EQ(any_status, std::future_status::ready);
  EXPECT_EQ(all_status, std::future_status::timeout);
  EXPECT_TRUE(retired.get());
  EXPECT_TRUE(any.get());
  EXPECT_TRUE(all.get());
  EXPECT_TRUE(assist_done->WaitHost());
  EXPECT_TRUE(device->WaitEvents({}));
  EXPECT_FALSE(device->WaitEvents({}, false));
}
}  // namespace

#ifdef LONGMARCH_D3D12_ENABLED
TEST(RhiTimeline, D3D12RetiresMainWhileAssistIsBlockedAndWaitsForAnyOrAll) {
  CheckIndependentRetirement(BACKEND_API_D3D12);
}
#endif
#ifdef LONGMARCH_VULKAN_ENABLED
TEST(RhiTimeline, VulkanRetiresMainWhileAssistIsBlockedAndWaitsForAnyOrAll) {
  CheckIndependentRetirement(BACKEND_API_VULKAN);
}
#endif
#ifdef LONGMARCH_METAL_ENABLED
TEST(RhiTimeline, MetalRetiresMainWhileAssistIsBlockedAndWaitsForAnyOrAll) {
  CheckIndependentRetirement(BACKEND_API_METAL);
}

TEST(RhiTimeline, MetalHostProgressTracksChainedGpuEvents) {
  ObjPtr<ICore> core = CreateCore(BACKEND_API_METAL);
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  ObjPtr<IQueue> queues[] = {device->GetQueue(QUEUE_TYPE_MAIN), device->GetQueue(QUEUE_TYPE_ASSIST)};
  ASSERT_TRUE(queues[0]);
  ASSERT_TRUE(queues[1]);
  std::vector<ObjPtr<ISubmissionToken>> submissions;
  Event previous{};
  for (int index = 0; index < 64; ++index) {
    ObjPtr<ICommandContext> commands = queues[index % 2]->CreateCommandContext();
    ASSERT_TRUE(commands);
    SubmitInfo info;
    if (previous.timeline)
      info.waits.push_back(previous);
    ObjPtr<ISubmissionToken> submitted = commands->Submit(info);
    ASSERT_TRUE(submitted);
    previous = submitted->GetEvent();
    submissions.push_back(std::move(submitted));
  }
  // The final GPU dependency transitively covers both queues. Host events must
  // also advance after completion, even when contexts are released immediately.
  ASSERT_TRUE(previous.timeline->WaitHost(previous.value));
  for (const auto &submitted : submissions) {
    const auto event = submitted->GetEvent();
    ASSERT_TRUE(event.timeline->WaitHost(event.value));
    uint64_t completed = 0;
    ASSERT_TRUE(event.timeline->GetCompletedValue(completed));
    EXPECT_GE(completed, event.value);
    EXPECT_TRUE(submitted->WaitHost());
  }
}
#endif

}  // namespace grassland::rhi
