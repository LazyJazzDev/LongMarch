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
static_assert(std::is_abstract_v<ISubmissionToken>);

class RhiRaw : public testing::TestWithParam<BackendAPI> {};

TEST_P(RhiRaw, PointerFactoriesAndBorrowedSubmissionWorkWithoutWrappers) {
  ICore *core = CreateCore(GetParam());
  ASSERT_NE(core, nullptr);
#if !defined(NDEBUG)
  EXPECT_EQ(core->CreateDevice(-1), nullptr);
#endif
  IDevice *device = core->CreateDeviceByAutoSelection();
  ASSERT_NE(device, nullptr);
  IQueue *queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  ASSERT_NE(queue, nullptr);
  ICommandContext *context = device->CreateCommandContext();
  ASSERT_NE(context, nullptr);
  ISubmissionToken *done = queue->Submit(context);
  ASSERT_NE(done, nullptr);
#if !defined(NDEBUG)
  EXPECT_FALSE(context->CmdClearImage(nullptr, {0, 0, 0, 1}));
  EXPECT_FALSE(context->CmdPresent(nullptr));
  EXPECT_EQ(queue->Submit(context), nullptr);  // One-shot submission.
#endif
  context->Release();  // Submit borrowed the caller reference and retained work.
  EXPECT_TRUE(done->WaitHost());
  EXPECT_EQ(done->GetStatus(), SUBMISSION_STATUS_COMPLETED);
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
#ifdef LONGMARCH_METAL_ENABLED
      BACKEND_API_METAL,
#endif
#ifdef LONGMARCH_VULKAN_ENABLED
      BACKEND_API_VULKAN,
#endif
  };
}

INSTANTIATE_TEST_SUITE_P(Backends, RhiRaw, testing::ValuesIn(Backends()));
}  // namespace
}  // namespace grassland::rhi
