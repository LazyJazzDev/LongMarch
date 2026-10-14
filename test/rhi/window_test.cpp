#include <gtest/gtest.h>

#include <cstdlib>

#include "grassland/rhi/backend/backend_window.h"
#include "grassland/rhi/rhix/rhix.h"

#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#undef CreateWindow
#endif

namespace grassland::rhix {
using namespace rhi;
namespace backend = rhi::backend;

namespace {

class RhiWindow : public testing::TestWithParam<BackendAPI> {};

TEST_P(RhiWindow, MultipleWindowsSingleUseFramesModesAndQueueLifetime) {
  if (!std::getenv("LONGMARCH_RHI_WINDOW_TESTS"))
    GTEST_SKIP() << "Set LONGMARCH_RHI_WINDOW_TESTS=1 on a desktop to run presentation tests.";
  auto core = rhix::CreateCore(GetParam());
  ASSERT_TRUE(core);
  auto device = core.CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  auto graphics = device.GetQueue(QueueType::Graphics);
  auto transfer = device.GetQueue(QueueType::Transfer);
  ASSERT_TRUE(graphics);
  ASSERT_TRUE(transfer);
  EXPECT_TRUE(graphics.SharesPhysicalQueue(transfer));
  auto first = device.CreateWindow({320, 240, "RHI window test A"});
  auto second = device.CreateWindow({320, 240, "RHI window test B"});
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  // Dropping an unsubmitted acquisition must not strand the swapchain.
  {
    auto discarded = first.AcquireFrame();
    ASSERT_TRUE(discarded);
  }
  SubmissionToken last;
  for (int i = 0; i < 12; ++i) {
    ASSERT_TRUE(Window::PollEvents());
    if (i == 4 || i == 8) {
      ASSERT_TRUE(first.SetPresentMode(i == 4 ? PresentMode::Continuous : PresentMode::VSync));
      if (i == 8)
        EXPECT_EQ(first.GetPresentMode(), PresentMode::VSync);
    }
    for (auto &window : {first, second}) {
      auto frame = window.AcquireFrame();
      ASSERT_TRUE(frame);
      EXPECT_GT(frame.GetDesc().width, 0u);
      auto context = device.CreateCommandContext(graphics);
      ASSERT_TRUE(context.ClearColor(frame, {0.6f, 0.7f, 0.8f, 1.0f}));
      ASSERT_TRUE(context.Present(frame));
      frame = {};  // Recording already owns the exact frame.
      last = graphics.Submit(context.Finish());
      ASSERT_TRUE(last);
    }
  }
  ASSERT_TRUE(last.WaitHost());
#ifdef _WIN32
  const auto hwnd = FindWindowA(nullptr, "RHI window test A");
  ASSERT_NE(hwnd, nullptr);
  auto old_frame = first.AcquireFrame();
  ASSERT_TRUE(old_frame);
  auto old_context = device.CreateCommandContext(graphics);
  ASSERT_TRUE(old_context.ClearColor(old_frame, {0, 0, 0, 1}));
  ASSERT_TRUE(old_context.Present(old_frame));
  auto old_packet = old_context.Finish();
  ASSERT_TRUE(MoveWindow(hwnd, 40, 40, 480, 360, TRUE));
  ASSERT_TRUE(Window::PollEvents());
  {
    auto resized = first.AcquireFrame();
    ASSERT_TRUE(resized);
  }
  // Holding this frame over resize violates the contract. We deliberately make
  // it invalid instead of allowing the old packet to target a new backbuffer.
  EXPECT_FALSE(old_frame);
  EXPECT_FALSE(graphics.Submit(std::move(old_packet)));
  old_frame = {};
  ShowWindow(hwnd, SW_MINIMIZE);
  ASSERT_TRUE(Window::PollEvents());
  EXPECT_FALSE(first.AcquireFrame());
  ASSERT_TRUE(first.SetPresentMode(PresentMode::Continuous));
  ShowWindow(hwnd, SW_RESTORE);
  ASSERT_TRUE(Window::PollEvents());
  {
    auto restored = first.AcquireFrame();
    ASSERT_TRUE(restored);
  }
#endif
  auto upload = device.CreateCommandContext(transfer);
  auto ready = transfer.Submit(upload.Finish());
  ASSERT_TRUE(ready);
  auto frame = second.AcquireFrame();
  ASSERT_TRUE(frame);
  auto context = device.CreateCommandContext(graphics);
  ASSERT_TRUE(context.ClearColor(frame, {0, 0, 1, 1}));
  ASSERT_TRUE(context.Present(frame));
  auto &window_backend = (*static_cast<backend::Window *>((second).Get()));
  frame.Release();
  first.Release();
  second.Release();
  // Only Device owns the Window. The recorded Image holds a frame lease,
  // not a strong Window reference, and keeps that lease until GPU completion.
  EXPECT_TRUE(window_backend.IsUniquelyOwned());
  EXPECT_TRUE(window_backend.HasFrameLeases());
  last = graphics.Submit(context.Finish(), {{ready}});
  ASSERT_TRUE(last);
  last.Release();
  ready.Release();
  graphics.Release();
  transfer.Release();
  device.Release();  // Device drains work after caller-owned children are released.
  core.Release();
  EXPECT_TRUE(Window::PollEvents());  // Main-thread GLFW destruction after retirement.
}

std::vector<BackendAPI> WindowBackends() {
  std::vector<BackendAPI> backends;
#ifdef LONGMARCH_D3D12_ENABLED
  backends.push_back(BACKEND_API_D3D12);
#endif
#ifdef LONGMARCH_VULKAN_ENABLED
  backends.push_back(BACKEND_API_VULKAN);
#endif
  return backends;
}

INSTANTIATE_TEST_SUITE_P(Backends, RhiWindow, testing::ValuesIn(WindowBackends()));

}  // namespace
}  // namespace grassland::rhix
