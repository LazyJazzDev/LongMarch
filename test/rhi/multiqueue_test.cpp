#include <gtest/gtest.h>

#include <chrono>
#include <future>
#include <type_traits>

#include "../../demo/rhi_multiqueue/experiment.h"
#include "grassland/rhi/backend/backend_command_context.h"
#include "grassland/rhi/backend/backend_timeline.h"
#include "grassland/rhi/rhix/rhix_event.h"
#if defined(LONGMARCH_D3D12_ENABLED)
#include "grassland/rhi/backend/d3d12/d3d12_device.h"
#include "grassland/rhi/backend/d3d12/d3d12_timeline.h"
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
#include "grassland/rhi/backend/vulkan/vulkan_device.h"
#include "grassland/rhi/backend/vulkan/vulkan_timeline.h"
#endif
#if defined(LONGMARCH_METAL_ENABLED)
#include "grassland/rhi/backend/metal/metal_device.h"
#include "grassland/rhi/backend/metal/metal_timeline.h"
#endif
namespace grassland::rhi {
namespace {
class Gate {
 public:
  bool Arm(IDevice *device, BackendAPI api) {
    device_ = device;
#if defined(LONGMARCH_D3D12_ENABLED)
    if (api == BACKEND_API_D3D12) {
      auto native = static_cast<backend::d3d12::D3D12Device *>(device);
      if (FAILED(native->NativeDevice()->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_))))
        return false;
      std::lock_guard<std::mutex> lock(native->QueueMutex(QUEUE_TYPE_TRANSFER));
      return SUCCEEDED(native->NativeQueue(QUEUE_TYPE_TRANSFER)->Wait(fence_.Get(), 1));
    }
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
    if (api == BACKEND_API_VULKAN) {
      auto native = static_cast<backend::vulkan::VulkanDevice *>(device);
      VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
      type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
      VkSemaphoreCreateInfo create{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
      create.pNext = &type;
      if (vkCreateSemaphore(native->NativeDevice(), &create, nullptr, &semaphore_) != VK_SUCCESS)
        return false;
      const uint64_t value = 1;
      VkTimelineSemaphoreSubmitInfo timeline{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
      timeline.waitSemaphoreValueCount = 1;
      timeline.pWaitSemaphoreValues = &value;
      VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
      VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
      submit.pNext = &timeline;
      submit.waitSemaphoreCount = 1;
      submit.pWaitSemaphores = &semaphore_;
      submit.pWaitDstStageMask = &stage;
      std::lock_guard<std::mutex> lock(native->QueueMutex(QUEUE_TYPE_TRANSFER));
      return vkQueueSubmit(native->NativeQueue(QUEUE_TYPE_TRANSFER), 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS;
    }
#endif
#if defined(LONGMARCH_METAL_ENABLED)
    if (api == BACKEND_API_METAL) {
      auto native = static_cast<backend::metal::MetalDevice *>(device);
      auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
      shared_ = NS::TransferPtr(native->NativeDevice()->newSharedEvent());
      if (!shared_)
        return false;
      auto command = native->CreateCommandBuffer(QUEUE_TYPE_TRANSFER);
      if (!command)
        return false;
      command->encodeWait(shared_.get(), 1);
      std::lock_guard<std::mutex> lock(native->QueueMutex(QUEUE_TYPE_TRANSFER));
      command->commit();
      return true;
    }
#endif
    return false;
  }

  void Release() {
    if (released_)
      return;
    released_ = true;
#if defined(LONGMARCH_D3D12_ENABLED)
    if (fence_)
      fence_->Signal(1);
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
    if (semaphore_) {
      auto native = static_cast<backend::vulkan::VulkanDevice *>(device_);
      VkSemaphoreSignalInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
      signal.semaphore = semaphore_;
      signal.value = 1;
      EXPECT_EQ(vkSignalSemaphore(native->NativeDevice(), &signal), VK_SUCCESS);
    }
#endif
#if defined(LONGMARCH_METAL_ENABLED)
    if (shared_)
      shared_->setSignaledValue(1);
#endif
  }

  ~Gate() {
    Release();
#if defined(LONGMARCH_D3D12_ENABLED)
    if (fence_) {
      auto native = static_cast<backend::d3d12::D3D12Device *>(device_);
      std::lock_guard<std::mutex> lock(native->QueueMutex(QUEUE_TYPE_TRANSFER));
      native->WaitIdle(QUEUE_TYPE_TRANSFER);
    }
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
    if (semaphore_) {
      auto native = static_cast<backend::vulkan::VulkanDevice *>(device_);
      // Also covers early assertion exits before a retirement token was created.
      vkQueueWaitIdle(native->NativeQueue(QUEUE_TYPE_TRANSFER));
      vkDestroySemaphore(native->NativeDevice(), semaphore_, nullptr);
    }
#endif
  }

 private:
  IDevice *device_{};
  bool released_{};
#if defined(LONGMARCH_D3D12_ENABLED)
  backend::d3d12::ComPtr<ID3D12Fence> fence_;
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
  VkSemaphore semaphore_{};
#endif
#if defined(LONGMARCH_METAL_ENABLED)
  NS::SharedPtr<MTL::SharedEvent> shared_;
#endif
};

class RhiMultiQueue : public testing::TestWithParam<BackendAPI> {};

TEST_P(RhiMultiQueue, ValueEventsUseTimelinesIndependentOfQueues) {
  static_assert(std::is_trivially_copyable_v<Event>);
  static_assert(std::is_same_v<Event, rhix::Event>);
  ObjPtr<ICore> core = CreateCore(GetParam(), {true});
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  auto create = [&]() -> backend::ObjectPtr<backend::Timeline> {
#if defined(LONGMARCH_D3D12_ENABLED)
    if (GetParam() == BACKEND_API_D3D12)
      return backend::ObjectPtr<backend::Timeline>::Adopt(
          backend::d3d12::D3D12Timeline::Create(static_cast<backend::d3d12::D3D12Device *>(device.Get())));
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
    if (GetParam() == BACKEND_API_VULKAN)
      return backend::ObjectPtr<backend::Timeline>::Adopt(
          backend::vulkan::VulkanTimeline::Create(static_cast<backend::vulkan::VulkanDevice *>(device.Get())));
#endif
#if defined(LONGMARCH_METAL_ENABLED)
    if (GetParam() == BACKEND_API_METAL)
      return backend::ObjectPtr<backend::Timeline>::Adopt(
          backend::metal::MetalTimeline::Create(static_cast<backend::metal::MetalDevice *>(device.Get())));
#endif
    return {};
  };
  auto signal = [&](ITimeline *timeline, uint64_t value) -> bool {
#if defined(LONGMARCH_D3D12_ENABLED)
    if (GetParam() == BACKEND_API_D3D12)
      return SUCCEEDED(static_cast<backend::d3d12::D3D12Timeline *>(timeline)->Native()->Signal(value));
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
    if (GetParam() == BACKEND_API_VULKAN) {
      VkSemaphoreSignalInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
      info.semaphore = static_cast<backend::vulkan::VulkanTimeline *>(timeline)->Native();
      info.value = value;
      return vkSignalSemaphore(static_cast<backend::vulkan::VulkanDevice *>(device.Get())->NativeDevice(), &info) ==
             VK_SUCCESS;
    }
#endif
#if defined(LONGMARCH_METAL_ENABLED)
    if (GetParam() == BACKEND_API_METAL) {
      static_cast<backend::metal::MetalTimeline *>(timeline)->Native()->setSignaledValue(value);
      return true;
    }
#endif
    return false;
  };
  auto first = create(), second = create(), third = create();
  ASSERT_TRUE(first && second && third);
  Event a{first.Get(), 2}, b{second.Get(), 4}, c{third.Get(), 6};
  Event copy = a;
  EXPECT_TRUE(first->IsUniquelyOwned());  // Copying Event never retains its timeline.
  EXPECT_EQ(copy.GetStatus(), SUBMISSION_STATUS_PENDING);
  ASSERT_TRUE(signal(a.timeline, a.value));
  ASSERT_TRUE(device->WaitEvents({a, b, c, {a.timeline, 100}}, false));
  ASSERT_TRUE(signal(b.timeline, b.value));
  ASSERT_TRUE(signal(c.timeline, c.value));
  ASSERT_TRUE(device->WaitEvents({a, b, c, {c.timeline, 1}}, true));
  uint64_t completed{};
  ASSERT_TRUE(a.timeline->GetCompletedValue(completed));
  EXPECT_EQ(completed, 2);
  EXPECT_TRUE(copy.WaitHost());
  EXPECT_FALSE(device->WaitEvents({{nullptr, 1}}));
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  ObjPtr<ICommandContext> commands = queue->CreateCommandContext();
  ASSERT_TRUE(commands);
  SubmitInfo info{{{a.timeline, 3}, b, c}};
  ObjPtr<ISubmissionToken> done = commands->Submit(info);
  ASSERT_TRUE(done);
  info.waits.clear();  // Native submission has consumed this storage and its values.
  a = {};
  ASSERT_TRUE(signal(first.Get(), 3));
  ASSERT_TRUE(done->WaitHost());
  Event finished = done->GetEvent();
  done.Release();
  EXPECT_TRUE(finished.WaitHost());
}

TEST_P(RhiMultiQueue, ContextKeepsItsPersistentQueueBindingAfterCallerHandlesAreReleased) {
  ObjPtr<ICore> core = CreateCore(GetParam(), {true});
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  for (auto type : {QUEUE_TYPE_GRAPHICS, QUEUE_TYPE_TRANSFER}) {
    ObjPtr<IQueue> first = device->GetQueue(type);
    ObjPtr<IQueue> second = device->GetQueue(type);
    ASSERT_TRUE(first && second);
    EXPECT_EQ(first.Get(), second.Get());
    auto identity = first.Get();
    ObjPtr<ICommandContext> context = first->CreateCommandContext();
    ASSERT_TRUE(context);
    EXPECT_EQ(static_cast<backend::CommandContext *>(context.Get())->OwnerQueue(), identity);
    first.Release();
    second.Release();
    ObjPtr<ISubmissionToken> done = context->Submit();
    ASSERT_TRUE(done);
    context.Release();
    ASSERT_TRUE(done->WaitHost());
    ObjPtr<IQueue> again = device->GetQueue(type);
    EXPECT_EQ(again.Get(), identity);
    EXPECT_EQ(done->GetStatus(), SUBMISSION_STATUS_COMPLETED);
  }
}

TEST_P(RhiMultiQueue, IndependentQueueCompletesAndRetiresWhileTransferIsBlocked) {
  ObjPtr<ICore> core = CreateCore(GetParam(), {true});
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  ObjPtr<IQueue> graphics = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  ObjPtr<IQueue> transfer = device->GetQueue(QUEUE_TYPE_TRANSFER);
  ASSERT_TRUE(graphics && transfer);
  if (graphics->SharesPhysicalQueue(transfer.Get()))
    GTEST_SKIP() << "Only one compatible native queue";
  EXPECT_TRUE(graphics->SharesPhysicalQueue(graphics.Get()));
  Gate gate;
  ASSERT_TRUE(gate.Arm(device.Get(), GetParam()));
  ObjPtr<ICommandContext> slow = transfer->CreateCommandContext();
  ObjPtr<ICommandContext> fast = graphics->CreateCommandContext();
  ASSERT_TRUE(slow && fast);
  ObjPtr<ISubmissionToken> blocked = slow->Submit();
  ObjPtr<ISubmissionToken> done = fast->Submit();
  ASSERT_TRUE(blocked && done);
  slow.Release();
  fast.Release();
  rhi::Event a = blocked->GetEvent();
  rhi::Event b = done->GetEvent();
  auto waiter = std::async(std::launch::async, [done] { return done->WaitHost(); });
  const bool retired_independently = waiter.wait_for(std::chrono::seconds(3)) == std::future_status::ready;
  const auto blocked_status = blocked->GetStatus();
  if (retired_independently) {
    EXPECT_TRUE(device->WaitEvents({a, b}, false));
    EXPECT_EQ(a.GetStatus(), SUBMISSION_STATUS_PENDING);
  }
  ObjPtr<ICommandContext> dependent_context = graphics->CreateCommandContext();
  if (!dependent_context) {
    gate.Release();
    FAIL() << "Could not create dependent context";
  }
  SubmitInfo dependencies;
  dependencies.waits = {a, a, b};
  ObjPtr<ISubmissionToken> dependent = dependent_context->Submit(dependencies);
  if (!dependent) {
    gate.Release();
    FAIL() << "Could not submit dependent context";
  }
  dependent_context.Release();
  auto dependent_waiter = std::async(std::launch::async, [dependent] { return dependent->WaitHost(); });
  const bool gpu_dependency_blocks =
      dependent_waiter.wait_for(std::chrono::milliseconds(50)) == std::future_status::timeout;
  gate.Release();
  ASSERT_TRUE(blocked->WaitHost());
  EXPECT_TRUE(waiter.get());
  EXPECT_TRUE(dependent_waiter.get());
  EXPECT_TRUE(gpu_dependency_blocks);
  EXPECT_TRUE(retired_independently);
  EXPECT_EQ(blocked_status, SUBMISSION_STATUS_PENDING);
  EXPECT_TRUE(device->WaitEvents({a, b, b}, true));
  done.Release();
  blocked.Release();
  EXPECT_TRUE(b.WaitHost());  // Immutable event outlives its SubmissionToken.
  EXPECT_EQ(a.GetStatus(), SUBMISSION_STATUS_COMPLETED);
}

TEST_P(RhiMultiQueue, EventDependencyMakesTransferredVertexColorsRenderCorrectly) {
  auto result = multiqueue_demo::Run(GetParam(), true, true, 8, 4);
  if (!result.distinct_queues && !result.device.empty())
    GTEST_SKIP() << "Only one compatible native queue";
  ASSERT_TRUE(result.success);
  EXPECT_TRUE(result.distinct_queues);
  EXPECT_EQ(result.frames, 8);
  EXPECT_EQ(result.mismatched_pixels, 0);
}

std::vector<BackendAPI> Backends() {
  std::vector<BackendAPI> result;
#if defined(LONGMARCH_D3D12_ENABLED)
  result.push_back(BACKEND_API_D3D12);
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
  result.push_back(BACKEND_API_VULKAN);
#endif
#if defined(LONGMARCH_METAL_ENABLED)
  result.push_back(BACKEND_API_METAL);
#endif
  return result;
}

INSTANTIATE_TEST_SUITE_P(Backends, RhiMultiQueue, testing::ValuesIn(Backends()));
}  // namespace
}  // namespace grassland::rhi
