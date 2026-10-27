#include <gtest/gtest.h>

#include "module.h"

namespace {
#if defined(TEST_RHIX_DEMO)
using CoreHandle = grassland::rhix::Core;
using DeviceHandle = grassland::rhix::Device;
#else
using CoreHandle = grassland::rhi::ObjPtr<grassland::rhi::ICore>;
using DeviceHandle = grassland::rhi::ObjPtr<grassland::rhi::IDevice>;
#endif

class ProbeModule final : public ModuleBase {
 public:
  using ModuleBase::ModuleBase;
  int initialization_result{}, update_result{}, render_result{}, termination_result{};
  int updates{}, renders{}, terminations{}, warmup{}, unavailable_frames{};
  bool close_during_update{}, multiple_windows{};
  std::vector<int> rendered_indices;

 private:
  int OnInit(const ModuleOptions &) override {
    updates = renders = terminations = 0;
    rendered_indices.clear();
    SetWarmupFrames(warmup);
    return initialization_result;
  }

  int OnUpdate() override {
    ++updates;
    // Closing the first of two windows must not terminate the module.
    if (close_during_update || (multiple_windows && updates == 4))
      RequestClose();
    return update_result;
  }

  int OnRender() override {
    ++renders;
    if (render_result)
      return render_result;
    if (renders <= unavailable_frames)
      return 0;
    rendered_indices.push_back(GetFrameIndex());
    CompleteFrame();
    return 0;
  }

  int OnTerminate() override {
    ++terminations;
    return termination_result;
  }
};

class DemoLifecycle : public testing::Test {
 protected:
  void SetUp() override {
#if defined(TEST_RHIX_DEMO)
    core = grassland::rhix::CreateCore();
    ASSERT_TRUE(core);
    device = core.CreateDeviceByAutoSelection();
#else
    core = grassland::rhi::CreateCore();
    ASSERT_TRUE(core);
    device = core->CreateDeviceByAutoSelection();
#endif
    ASSERT_TRUE(device);
  }

  CoreHandle core;
  DeviceHandle device;
};

TEST_F(DemoLifecycle, WarmupAndUnavailableFramesDoNotConsumeFrameBudget) {
  ProbeModule module(core, device);
  module.warmup = 2;
  module.unavailable_frames = 2;
  ModuleOptions options;
  options.headless = true;
  options.frames = 2;
  EXPECT_EQ(module.Run(options), 0);
  EXPECT_EQ(module.rendered_indices, (std::vector<int>{-2, -1, 0, 1}));
  EXPECT_EQ(module.updates, 6);
  EXPECT_EQ(module.terminations, 1);
  EXPECT_TRUE(module.ShouldClose());
  EXPECT_EQ(module.Run(options), 0);
  EXPECT_EQ(module.updates, 6);
  EXPECT_EQ(module.terminations, 1);
}

TEST_F(DemoLifecycle, ClosingDuringUpdatePreventsRendering) {
  ProbeModule module(core, device);
  module.close_during_update = true;
  ModuleOptions options;
  options.headless = true;
  EXPECT_EQ(module.Run(options), 0);
  EXPECT_EQ(module.updates, 1);
  EXPECT_EQ(module.renders, 0);
  EXPECT_EQ(module.terminations, 1);
}

TEST_F(DemoLifecycle, ModuleLifetimeIsIndependentOfIndividualWindows) {
  ProbeModule module(core, device);
  module.multiple_windows = true;
  ModuleOptions options;
  options.headless = true;
  EXPECT_EQ(module.Run(options), 0);
  EXPECT_EQ(module.renders, 3);
  EXPECT_EQ(module.terminations, 1);
}

TEST_F(DemoLifecycle, EveryFailurePathTerminatesAndPreservesTheFirstError) {
  ModuleOptions options;
  options.headless = true;
  options.frames = 1;
  for (int stage = 0; stage != 4; ++stage) {
    ProbeModule module(core, device);
    if (stage == 0)
      module.initialization_result = 7;
    if (stage == 1)
      module.update_result = 7;
    if (stage == 2)
      module.render_result = 7;
    module.termination_result = 9;
    EXPECT_EQ(module.Run(options), stage == 3 ? 9 : 7);
    EXPECT_EQ(module.terminations, 1);
    EXPECT_TRUE(module.ShouldClose());
    if (stage < 2)
      EXPECT_EQ(module.renders, 0);
  }
}
}  // namespace
