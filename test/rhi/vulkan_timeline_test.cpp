#include <gtest/gtest.h>

#include <array>
#include <future>

#include "grassland/rhi/backend/vulkan/vulkan_queue.h"
#include "grassland/rhi/rhi.h"

namespace grassland::rhi::backend::vulkan {
namespace {

TEST(RhiVulkanTimeline, ConcurrentLogicalQueueSubmissionsKeepSeparateTimelines) {
  ObjPtr<ICore> core = CreateCore(BACKEND_API_VULKAN, {true});
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  std::array<ObjPtr<IQueue>, 2> queues{device->GetQueue(QUEUE_TYPE_MAIN), device->GetQueue(QUEUE_TYPE_ASSIST)};
  ASSERT_TRUE(queues[0]);
  ASSERT_TRUE(queues[1]);
  EXPECT_NE(queues[0].Get(), queues[1].Get());
  auto submit = [](IQueue *queue) {
    std::vector<ObjPtr<ISubmissionToken>> tokens;
    for (uint64_t serial = 1; serial <= 128; ++serial) {
      ObjPtr<ICommandContext> commands = queue->CreateCommandContext();
      if (!commands)
        return false;
      ObjPtr<ISubmissionToken> token = commands->Submit();
      if (!token || token->GetEvent().value != serial ||
          token->GetEvent().timeline != static_cast<IVulkanQueue *>(queue)->GetTimeline())
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
  EXPECT_TRUE(main.get());
  EXPECT_TRUE(assist.get());
}

TEST(RhiVulkanTimeline, CompletionValuesSurviveContextRecyclingAcrossQueues) {
  ObjPtr<ICore> core = CreateCore(BACKEND_API_VULKAN);
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  std::array<ObjPtr<IQueue>, 2> queues{device->GetQueue(QUEUE_TYPE_MAIN), device->GetQueue(QUEUE_TYPE_ASSIST)};
  ASSERT_TRUE(queues[0]);
  ASSERT_TRUE(queues[1]);
  std::array<Event, 2> first{};
  for (const auto &queue : queues) {
    auto native = static_cast<IVulkanQueue *>(queue.Get());
    EXPECT_EQ(native->Wait(0, 0), VK_SUCCESS);
    EXPECT_EQ(native->Wait(1, 0), VK_TIMEOUT);
  }
  for (int batch = 0; batch < 8; ++batch) {
    std::vector<ObjPtr<ISubmissionToken>> tokens;
    for (int i = 0; i < 8; ++i) {
      const auto index = i % 2;
      ObjPtr<ICommandContext> context = queues[index]->CreateCommandContext();
      ASSERT_TRUE(context);
      ObjPtr<ISubmissionToken> token = context->Submit();
      ASSERT_TRUE(token);
      const auto event = token->GetEvent();
      EXPECT_EQ(event.timeline, static_cast<IVulkanQueue *>(queues[index].Get())->GetTimeline());
      EXPECT_EQ(event.value, uint64_t(batch * 4 + i / 2 + 1));
      if (!first[index].timeline)
        first[index] = event;
      tokens.push_back(std::move(token));
    }
    for (const auto &token : tokens)
      ASSERT_TRUE(token->WaitHost());
    for (size_t index = 0; index < queues.size(); ++index) {
      uint64_t completed = 0;
      ASSERT_TRUE(static_cast<IVulkanQueue *>(queues[index].Get())->GetTimeline()->GetCompletedValue(completed));
      EXPECT_EQ(completed, uint64_t((batch + 1) * 4));
      EXPECT_TRUE(first[index].timeline->WaitHost(first[index].value));
    }
  }
}

}  // namespace
}  // namespace grassland::rhi::backend::vulkan
