#include <gtest/gtest.h>

#include "grassland/rhi/rhi.h"
#include "sparkium/core/resource_updates.h"
#include "sparkium/pipelines/raytracing/core/core.h"

namespace {
using namespace grassland;
using namespace grassland::rhi;

class SparkiumBindingTest : public testing::TestWithParam<BackendAPI> {
 protected:
  void SetUp() override {
    core_ = CreateCore(GetParam());
    ASSERT_TRUE(core_);
    device_ = core_->CreateDeviceByAutoSelection();
    ASSERT_TRUE(device_);
  }

  ObjPtr<ICore> core_;
  ObjPtr<IDevice> device_;
};

TEST_P(SparkiumBindingTest, InvalidGeometryLightShaderReturnsFailureWithoutThrowing) {
  auto core = sparkium::Core::Create(device_.Get());
  ASSERT_TRUE(core);
  auto *tracing = sparkium::raytracing::DedicatedCast(core.get());
  ASSERT_TRUE(tracing);
  EXPECT_EQ(tracing->GetGeometryLightPowerProgram("#error invalid geometry for failure regression", ""), nullptr);
  EXPECT_FALSE(core->IsValid());
}

TEST_P(SparkiumBindingTest, UploadSnapshotsSurviveCanceledRecordingAndAcceptOnlyRecordedPrefix) {
  sparkium::ResourceUpdates updates;
  ObjPtr<IBuffer> buffer = device_->CreateBuffer({sizeof(uint32_t)});
  ObjPtr<IQueue> queue = device_->GetQueue(QUEUE_TYPE_ASSIST);
  ASSERT_TRUE(buffer);
  ASSERT_TRUE(queue);
  int accepted = 0;
  uint32_t source = 17;
  ASSERT_TRUE(updates.Upload(buffer.Get(), &source, sizeof(source)));
  source = 29;
  {
    ObjPtr<ICommandContext> canceled = queue->CreateCommandContext();
    auto cursor = updates.Begin();
    cursor.OnAccepted([&] { ++accepted; });
    ASSERT_TRUE(updates.Record(canceled.Get(), cursor));
  }
  EXPECT_EQ(accepted, 0);
  auto cursor = updates.Begin();
  cursor.OnAccepted([&] { ++accepted; });
  ObjPtr<ICommandContext> commands = queue->CreateCommandContext();
  ASSERT_TRUE(updates.Record(commands.Get(), cursor));
  ASSERT_TRUE(updates.Upload(buffer.Get(), &source, sizeof(source)));
  ObjPtr<IDownloadToken> first = commands->CmdDownloadBuffer(buffer.Get(), 0, sizeof(source));
  ASSERT_TRUE(first);
  ObjPtr<ISubmissionToken> submitted = commands->Submit();
  ASSERT_TRUE(submitted);
  ASSERT_TRUE(updates.Accept(cursor));
  EXPECT_FALSE(updates.Accept(cursor));
  EXPECT_EQ(accepted, 1);
  EXPECT_FALSE(updates.Empty());
  ASSERT_TRUE(submitted->WaitHost());
  uint32_t actual{};
  ASSERT_TRUE(first->CopyTo(&actual, sizeof(actual)));
  EXPECT_EQ(actual, 17u);

  cursor = updates.Begin();
  commands = queue->CreateCommandContext();
  ASSERT_TRUE(updates.Record(commands.Get(), cursor));
  ObjPtr<IDownloadToken> second = commands->CmdDownloadBuffer(buffer.Get(), 0, sizeof(source));
  ASSERT_TRUE(second);
  submitted = commands->Submit();
  ASSERT_TRUE(submitted);
  ASSERT_TRUE(updates.Accept(cursor));
  EXPECT_TRUE(updates.Empty());
  ASSERT_TRUE(submitted->WaitHost());
  ASSERT_TRUE(second->CopyTo(&actual, sizeof(actual)));
  EXPECT_EQ(actual, 29u);
}

const std::vector<BackendAPI> backends{
#if defined(LONGMARCH_VULKAN_ENABLED)
    BACKEND_API_VULKAN,
#endif
#if defined(LONGMARCH_D3D12_ENABLED)
    BACKEND_API_D3D12,
#endif
#if defined(LONGMARCH_METAL_ENABLED)
    BACKEND_API_METAL,
#endif
};
INSTANTIATE_TEST_SUITE_P(Backends, SparkiumBindingTest, testing::ValuesIn(backends));
}  // namespace
