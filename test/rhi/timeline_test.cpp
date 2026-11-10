#include <gtest/gtest.h>

#include "grassland/rhi/rhi.h"

#ifdef LONGMARCH_VULKAN_ENABLED
#include "grassland/rhi/backend/vulkan/vulkan_device.h"
#include "grassland/rhi/backend/vulkan/vulkan_timeline.h"
#endif
#ifdef LONGMARCH_D3D12_ENABLED
#include "grassland/rhi/backend/d3d12/d3d12_device.h"
#include "grassland/rhi/backend/d3d12/d3d12_timeline.h"
#endif

namespace grassland::rhi {

#ifdef LONGMARCH_VULKAN_ENABLED
TEST(RhiTimeline, VulkanIndependentTimelineTracksGPUSignals) {
  using namespace backend::vulkan;
  ObjPtr<ICore> core = CreateCore(BACKEND_API_VULKAN);
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  auto native = static_cast<VulkanDevice *>(device.Get());
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
