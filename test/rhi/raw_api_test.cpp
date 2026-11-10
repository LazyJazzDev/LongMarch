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

TEST_P(RhiRaw, QueueIdentitySurvivesCallerReferenceRelease) {
  ObjPtr<ICore> core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  IQueue *identities[2]{};
  for (auto type : {QUEUE_TYPE_MAIN, QUEUE_TYPE_ASSIST}) {
    ObjPtr<IQueue> queue = device->GetQueue(type);
    ASSERT_TRUE(queue);
    ObjPtr<IQueue> second = device->GetQueue(type);
    EXPECT_EQ(queue.Get(), second.Get());
    identities[type] = queue.Get();
  }
  EXPECT_NE(identities[0], identities[1]);
  for (auto type : {QUEUE_TYPE_MAIN, QUEUE_TYPE_ASSIST}) {
    ObjPtr<IQueue> queue = device->GetQueue(type);
    ASSERT_TRUE(queue);
    EXPECT_EQ(queue.Get(), identities[type]);
    EXPECT_EQ(queue->GetType(), type);
  }
}

TEST_P(RhiRaw, PointerFactoriesAndBorrowedSubmissionWorkWithoutWrappers) {
  ICore *core = CreateCore(GetParam());
  ASSERT_NE(core, nullptr);
#if !defined(NDEBUG)
  EXPECT_EQ(core->CreateDevice(-1), nullptr);
#endif
  IDevice *device = core->CreateDeviceByAutoSelection();
  ASSERT_NE(device, nullptr);
  IQueue *queue = device->GetQueue(QUEUE_TYPE_MAIN);
  ASSERT_NE(queue, nullptr);
  ICommandContext *context = ObjPtr<IQueue>(device->GetQueue(QUEUE_TYPE_MAIN))->CreateCommandContext();
  ASSERT_NE(context, nullptr);
  ISubmissionToken *done = context->Submit();
  ASSERT_NE(done, nullptr);
#if !defined(NDEBUG)
  EXPECT_FALSE(context->CmdClearImage(nullptr, {0.0f, 0.0f, 0.0f, 1.0f}));
  EXPECT_FALSE(context->CmdPresent(nullptr));
  EXPECT_EQ(context->Submit(), nullptr);  // One-shot submission.
#endif
  context->Release();  // Submit borrowed the caller reference and retained work.
  EXPECT_TRUE(done->WaitHost());
  EXPECT_EQ(done->GetStatus(), SUBMISSION_STATUS_COMPLETED);
  done->Release();
  queue->Release();
  device->Release();
  core->Release();
}

TEST_P(RhiRaw, ContextSubmitsAfterCallerReleasesItsQueueReference) {
  ObjPtr<ICore> core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  for (auto type : {QUEUE_TYPE_MAIN, QUEUE_TYPE_ASSIST}) {
    ObjPtr<IQueue> queue = device->GetQueue(type);
    ASSERT_TRUE(queue);
    ObjPtr<ICommandContext> commands = queue->CreateCommandContext();
    ASSERT_TRUE(commands);
    queue.Release();
    ObjPtr<ISubmissionToken> done = commands->Submit();
    ASSERT_TRUE(done);
    commands.Release();
    EXPECT_TRUE(done->WaitHost());
  }
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
