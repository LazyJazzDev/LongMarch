#include <gtest/gtest.h>

#include "grassland/rhi/backend/vulkan/vulkan_device.h"
#include "grassland/rhi/rhi.h"

namespace grassland::rhi::backend::vulkan {
namespace {

TEST(RhiVulkanTimeline, CompletionValuesSurviveContextRecyclingAcrossLogicalQueues) {
  ObjPtr<ICore> core = CreateCore(BACKEND_API_VULKAN);
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  auto native = static_cast<VulkanDevice *>(device.Get());
  const auto timeline = native->Timeline();
  ASSERT_NE(timeline, VK_NULL_HANDLE);
  EXPECT_EQ(native->Wait(0, 0), VK_SUCCESS);
  EXPECT_EQ(native->Wait(1, 0), VK_TIMEOUT);
  ObjPtr<IQueue> graphics = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  ObjPtr<IQueue> transfer = device->GetQueue(QUEUE_TYPE_TRANSFER);
  ASSERT_TRUE(graphics);
  ASSERT_TRUE(transfer);
  uint64_t previous = 0;
  uint64_t first = 0;
  for (int batch = 0; batch < 8; ++batch) {
    std::vector<ObjPtr<ISubmissionToken>> tokens;
    for (int i = 0; i < 8; ++i) {
      ObjPtr<ICommandContext> context = device->CreateCommandContext();
      ASSERT_TRUE(context);
      auto queue = i % 2 ? graphics.Get() : transfer.Get();
      ObjPtr<ISubmissionToken> token = queue->Submit(context.Get());
      ASSERT_TRUE(token);
      tokens.push_back(std::move(token));
    }  // Caller references are gone; Device retires and recycles native commands.
    for (const auto &token : tokens)
      ASSERT_TRUE(token->WaitHost());
    uint64_t completed = 0;
    ASSERT_EQ(vkGetSemaphoreCounterValue(native->NativeDevice(), timeline, &completed), VK_SUCCESS);
    EXPECT_EQ(completed, previous + 8);
    if (!first)
      first = completed;
    EXPECT_EQ(native->Wait(first, 0), VK_SUCCESS);  // Old values never lose their meaning on reuse.
    previous = completed;
  }
}

}  // namespace
}  // namespace grassland::rhi::backend::vulkan
