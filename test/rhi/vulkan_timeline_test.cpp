#include <gtest/gtest.h>

#include <array>
#include <future>

#include "grassland/rhi/backend/vulkan/vulkan_queue.h"
#include "grassland/rhi/rhi.h"

namespace grassland::rhi::backend::vulkan {
namespace {

TEST(RhiVulkanQueues, SelectsDedicatedComputeAndFallsBackToSharedQueues) {
  const VkQueueFlags graphics = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;

  struct Case {
    uint32_t graphics_count, dedicated_count, requested_count;
    std::array<uint32_t, 3> indices;
  };

  for (const auto &test :
       {Case{1, 0, 1, {0, 0, 0}}, Case{2, 0, 2, {0, 1, 1}}, Case{3, 0, 3, {0, 1, 2}}, Case{2, 1, 2, {0, 1, 0}}}) {
    SCOPED_TRACE(test.graphics_count);
    SCOPED_TRACE(test.dedicated_count);
    std::vector<VkQueueFamilyProperties> families(2);
    families[0].queueFlags = graphics;
    families[0].queueCount = test.graphics_count;
    families[1].queueFlags = VK_QUEUE_COMPUTE_BIT;
    families[1].queueCount = test.dedicated_count;
    const auto selected = SelectQueues(families);
    EXPECT_EQ(selected.graphics_family, 0u);
    EXPECT_EQ(selected.compute_family, test.dedicated_count ? 1u : 0u);
    EXPECT_EQ(selected.graphics_count, test.requested_count);
    for (size_t i = 0; i < test.indices.size(); ++i)
      EXPECT_EQ(selected.indices[i], test.indices[i]);
  }
  EXPECT_EQ(SelectQueues({}).graphics_family, UINT32_MAX);
}

TEST(RhiVulkanTimeline, ConcurrentLogicalQueueSubmissionsKeepSeparateTimelines) {
  ObjPtr<ICore> core = CreateCore(BACKEND_API_VULKAN, {true});
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  std::array<ObjPtr<IQueue>, 3> queues{device->GetQueue(QUEUE_TYPE_MAIN), device->GetQueue(QUEUE_TYPE_ASSIST),
                                       device->GetQueue(QUEUE_TYPE_COMPUTE)};
  ASSERT_TRUE(queues[0]);
  ASSERT_TRUE(queues[1]);
  ASSERT_TRUE(queues[2]);
  EXPECT_NE(queues[0].Get(), queues[1].Get());
  auto submit = [](IQueue *queue) {
    std::vector<ObjPtr<ISubmissionToken>> tokens;
    for (uint64_t serial = 1; serial <= 128; ++serial) {
      ObjPtr<ICommandContext> commands = queue->CreateCommandContext();
      if (!commands)
        return false;
      ObjPtr<ISubmissionToken> token = commands->Submit();
      if (!token || token->GetEvent().value != serial ||
          token->GetEvent().timeline !=
              rhi::QueryInterfacePtr<ITimeline>(rhi::QueryInterfacePtr<IVulkanQueue>(queue)->GetTimeline()).Get())
        return false;
      tokens.push_back(std::move(token));
    }
    for (const auto &token : tokens)
      if (!token->WaitHost())
        return false;
    return true;
  };
  auto main = std::async(std::launch::async, submit, queues[0].Get());
  auto assist = std::async(std::launch::async, submit, queues[1].Get());
  auto compute = std::async(std::launch::async, submit, queues[2].Get());
  EXPECT_TRUE(compute.get());
  EXPECT_TRUE(main.get());
  EXPECT_TRUE(assist.get());
}

TEST(RhiVulkanTimeline, CompletionValuesSurviveContextRecyclingAcrossQueues) {
  ObjPtr<ICore> core = CreateCore(BACKEND_API_VULKAN);
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  std::array<ObjPtr<IQueue>, 3> queues{device->GetQueue(QUEUE_TYPE_MAIN), device->GetQueue(QUEUE_TYPE_ASSIST),
                                       device->GetQueue(QUEUE_TYPE_COMPUTE)};
  ASSERT_TRUE(queues[0]);
  ASSERT_TRUE(queues[1]);
  ASSERT_TRUE(queues[2]);
  std::array<Event, 3> first{};
  for (const auto &queue : queues) {
    auto native = rhi::QueryInterfacePtr<IVulkanQueue>(queue.Get()).Get();
    EXPECT_EQ(native->Wait(0, 0), VK_SUCCESS);
    EXPECT_EQ(native->Wait(1, 0), VK_TIMEOUT);
  }
  for (int batch = 0; batch < 8; ++batch) {
    std::vector<ObjPtr<ISubmissionToken>> tokens;
    for (int i = 0; i < 12; ++i) {
      const auto index = i % 3;
      ObjPtr<ICommandContext> context = queues[index]->CreateCommandContext();
      ASSERT_TRUE(context);
      ObjPtr<ISubmissionToken> token = context->Submit();
      ASSERT_TRUE(token);
      const auto event = token->GetEvent();
      EXPECT_EQ(event.timeline, rhi::QueryInterfacePtr<ITimeline>(
                                    rhi::QueryInterfacePtr<IVulkanQueue>(queues[index].Get())->GetTimeline())
                                    .Get());
      EXPECT_EQ(event.value, uint64_t(batch * 4 + i / 3 + 1));
      if (!first[index].timeline)
        first[index] = event;
      tokens.push_back(std::move(token));
    }
    for (const auto &token : tokens)
      ASSERT_TRUE(token->WaitHost());
    for (size_t index = 0; index < queues.size(); ++index) {
      uint64_t completed = 0;
      ASSERT_TRUE(
          rhi::QueryInterfacePtr<ITimeline>(rhi::QueryInterfacePtr<IVulkanQueue>(queues[index].Get())->GetTimeline())
              .Get()
              ->GetCompletedValue(completed));
      EXPECT_EQ(completed, uint64_t((batch + 1) * 4));
      EXPECT_TRUE(first[index].timeline->WaitHost(first[index].value));
    }
  }
}

}  // namespace
}  // namespace grassland::rhi::backend::vulkan
