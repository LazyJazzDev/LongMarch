#include <gtest/gtest.h>

#include <type_traits>

// This translation unit deliberately needs neither rhix nor backend headers.
#include "grassland/rhi/rhi.h"

namespace grassland::rhi {
namespace {
static_assert(std::is_abstract_v<ICore>);
static_assert(std::is_abstract_v<IDevice>);
static_assert(std::is_abstract_v<IWindow>);
static_assert(std::is_abstract_v<IImage>);
static_assert(std::is_abstract_v<IQueue>);
static_assert(std::is_abstract_v<ICommandContext>);
static_assert(std::is_abstract_v<IRecordedCommands>);
static_assert(std::is_abstract_v<ISubmissionToken>);

class RhiRaw : public testing::TestWithParam<BackendAPI> {};

TEST_P(RhiRaw, PointerFactoriesAndBorrowedSubmissionWorkWithoutWrappers) {
  ICore *core = CreateCore(GetParam());
  ASSERT_NE(core, nullptr);
  EXPECT_EQ(core->CreateDevice(-1), nullptr);
  IDevice *device = core->CreateDeviceByAutoSelection();
  ASSERT_NE(device, nullptr);
  IQueue *queue = device->GetQueue(QueueType::Graphics);
  ASSERT_NE(queue, nullptr);
  ICommandContext *context = device->CreateCommandContext(queue);
  ASSERT_NE(context, nullptr);
  // A recording is not a finished packet, even though it shares the interface.
  EXPECT_EQ(queue->Submit(context), nullptr);
  IRecordedCommands *commands = context->Finish();
  ASSERT_NE(commands, nullptr);
  EXPECT_FALSE(context->ClearColor(nullptr, {0, 0, 0, 1}));
  EXPECT_EQ(context->Finish(), nullptr);
  context->Release();  // Finish returned its own reference.
  ISubmissionToken *done = queue->Submit(commands);
  ASSERT_NE(done, nullptr);
  EXPECT_EQ(queue->Submit(commands), nullptr);  // One-shot submission.
  commands->Release();                          // Submit borrowed the caller reference and retained work.
  EXPECT_TRUE(done->WaitHost());
  EXPECT_EQ(done->GetStatus(), SubmissionStatus::Completed);
  done->Release();
  queue->Release();
  device->Release();
  core->Release();
}

std::vector<BackendAPI> Backends() {
  return {
#ifdef LONGMARCH_D3D12_ENABLED
      BACKEND_API_D3D12,
#endif
#ifdef LONGMARCH_VULKAN_ENABLED
      BACKEND_API_VULKAN,
#endif
  };
}

INSTANTIATE_TEST_SUITE_P(Backends, RhiRaw, testing::ValuesIn(Backends()));
}  // namespace
}  // namespace grassland::rhi
