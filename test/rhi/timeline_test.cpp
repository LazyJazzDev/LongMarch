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
#include "grassland/rhi/backend/d3d12/d3d12_queue.h"
#include "grassland/rhi/backend/d3d12/d3d12_timeline.h"
#endif

#ifdef LONGMARCH_METAL_ENABLED
#include "grassland/rhi/backend/metal/metal_device.h"
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
  std::unique_ptr<backend::Timeline> gate;

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
    auto timeline = backend::d3d12::D3D12Timeline::Create(static_cast<backend::d3d12::D3D12Device *>(device.Get()));
    ASSERT_TRUE(timeline);
    auto fence = timeline->NativeFence();
    release.open = [fence] { EXPECT_EQ(fence->Signal(1), S_OK); };
    gate = std::move(timeline);
  }
#endif
#ifdef LONGMARCH_VULKAN_ENABLED
  if (api == BACKEND_API_VULKAN) {
    auto native = static_cast<backend::vulkan::VulkanDevice *>(device.Get());
    auto timeline = backend::vulkan::VulkanTimeline::Create(native);
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
    auto timeline = backend::metal::MetalTimeline::Create(static_cast<backend::metal::MetalDevice *>(device.Get()));
    ASSERT_TRUE(timeline);
    auto event = timeline->NativeEvent();
    release.open = [event] { event->setSignaledValue(1); };
    gate = std::move(timeline);
  }
#endif
  ASSERT_TRUE(gate);
  ObjPtr<ICommandContext> blocked = assist->CreateCommandContext();
  ASSERT_TRUE(blocked);
  ObjPtr<ISubmissionToken> assist_done = blocked->Submit({{{gate.get(), 1}}});
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
#endif

#ifdef LONGMARCH_VULKAN_ENABLED
TEST(RhiTimeline, VulkanIndependentTimelineTracksGPUSignals) {
  using namespace backend::vulkan;
  ObjPtr<ICore> core = CreateCore(BACKEND_API_VULKAN);
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  auto native = static_cast<VulkanDevice *>(device.Get());
  ObjPtr<IQueue> main = device->GetQueue(QUEUE_TYPE_MAIN);
  ObjPtr<IQueue> assist = device->GetQueue(QUEUE_TYPE_ASSIST);
  ASSERT_TRUE(main);
  ASSERT_TRUE(assist);
  auto main_queue = static_cast<VulkanQueue *>(main.Get());
  auto assist_queue = static_cast<VulkanQueue *>(assist.Get());
  EXPECT_NE(main_queue->NativeQueue(), assist_queue->NativeQueue());
  EXPECT_NE(main_queue->GetTimeline(), assist_queue->GetTimeline());
  EXPECT_FALSE(main->SharesPhysicalQueue(assist.Get()));
  auto timeline = VulkanTimeline::Create(native);
  ASSERT_TRUE(timeline);
  uint64_t completed = UINT64_MAX;
  ASSERT_TRUE(timeline->GetCompletedValue(completed));
  EXPECT_EQ(completed, 0);
  ASSERT_TRUE(timeline->WaitHost(0));
  const auto semaphore = timeline->NativeSemaphore();
  for (uint64_t value : {7, 12}) {
    VkTimelineSemaphoreSubmitInfo values{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    values.signalSemaphoreValueCount = 1;
    values.pSignalSemaphoreValues = &value;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.pNext = &values;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &semaphore;
    {
      std::lock_guard<std::mutex> lock(native->QueueMutex());
      ASSERT_EQ(vkQueueSubmit(native->NativeQueue(), 1, &submit, VK_NULL_HANDLE), VK_SUCCESS);
    }
    ASSERT_TRUE(timeline->WaitHost(value));
    ASSERT_TRUE(timeline->GetCompletedValue(completed));
    EXPECT_EQ(completed, value);
  }
  EXPECT_TRUE(timeline->WaitHost(7));
}
#endif

#ifdef LONGMARCH_D3D12_ENABLED
TEST(RhiTimeline, D3D12IndependentTimelineTracksGPUSignals) {
  using namespace backend::d3d12;
  ObjPtr<ICore> core = CreateCore(BACKEND_API_D3D12);
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  auto native = static_cast<D3D12Device *>(device.Get());
  ObjPtr<IQueue> main = device->GetQueue(QUEUE_TYPE_MAIN);
  ObjPtr<IQueue> assist = device->GetQueue(QUEUE_TYPE_ASSIST);
  ASSERT_TRUE(main);
  ASSERT_TRUE(assist);
  auto main_queue = static_cast<D3D12Queue *>(main.Get());
  auto assist_queue = static_cast<D3D12Queue *>(assist.Get());
  EXPECT_NE(main_queue->NativeQueue(), assist_queue->NativeQueue());
  EXPECT_NE(main_queue->GetTimeline(), assist_queue->GetTimeline());
  EXPECT_FALSE(main->SharesPhysicalQueue(assist.Get()));
  auto timeline = D3D12Timeline::Create(native);
  ASSERT_TRUE(timeline);
  uint64_t completed = UINT64_MAX;
  ASSERT_TRUE(timeline->GetCompletedValue(completed));
  EXPECT_EQ(completed, 0);
  ASSERT_TRUE(timeline->WaitHost(0));
  for (uint64_t value : {7, 12}) {
    {
      std::lock_guard<std::mutex> lock(native->QueueMutex());
      ASSERT_EQ(native->NativeQueue()->Signal(timeline->NativeFence(), value), S_OK);
    }
    ASSERT_TRUE(timeline->WaitHost(value));
    ASSERT_TRUE(timeline->GetCompletedValue(completed));
    EXPECT_EQ(completed, value);
  }
  EXPECT_TRUE(timeline->WaitHost(7));
}
#endif

}  // namespace grassland::rhi
