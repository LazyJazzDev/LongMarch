#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>

#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#endif

#include "grassland/rhi/backend/backend_window.h"
#include "grassland/rhi/rhix/rhix.h"

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
  auto graphics = device.GetQueue(QUEUE_TYPE_GRAPHICS);
  auto transfer = device.GetQueue(QUEUE_TYPE_TRANSFER);
  ASSERT_TRUE(graphics);
  ASSERT_TRUE(transfer);
  EXPECT_TRUE(graphics.SharesPhysicalQueue(transfer));
  auto first = device.CreateGLFWWindow({320, 240, "RHI window test A"});
  auto second = device.CreateGLFWWindow({320, 240, "RHI window test B"});
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  {
    auto frame = first.AcquireFrame();
    ASSERT_TRUE(frame);
    auto context = device.CreateCommandContext(graphics);
    ASSERT_TRUE(context.CmdClearColor(frame, {0, 0, 0, 1}));
    EXPECT_FALSE(context.CmdPresent(second));
    EXPECT_FALSE(graphics.Submit(std::move(context)));
  }
  ASSERT_NE(first.GLFWWindow(), nullptr);
  EXPECT_EQ(first.GLFWWindow(), first.Get()->GLFWWindow());
  EXPECT_EQ(first.IsKeyDown(GLFW_KEY_A), glfwGetKey(first.GLFWWindow(), GLFW_KEY_A) == GLFW_PRESS);
  glfwSetWindowShouldClose(first.GLFWWindow(), GLFW_TRUE);
  EXPECT_TRUE(first.ShouldClose());
  glfwSetWindowShouldClose(first.GLFWWindow(), GLFW_FALSE);
  EXPECT_FALSE(first.ShouldClose());
  const auto options = first.GetSupportedPresentationOptions();
  ASSERT_FALSE(options.empty());
  for (size_t i = 0; i < options.size(); ++i) {
    EXPECT_NE(options[i].format, IMAGE_FORMAT_UNKNOWN);
    for (size_t j = 0; j < i; ++j)
      EXPECT_FALSE(options[i].format == options[j].format && options[i].present_mode == options[j].present_mode);
    ASSERT_TRUE(first.SetPresentMode(options[i].present_mode));
    EXPECT_EQ(first.GetPresentMode(), options[i].present_mode);
  }
  ASSERT_TRUE(first.SetPresentMode(PRESENT_MODE_VSYNC));
  // Dropping a Vulkan acquisition requires explicit recovery, never an implicit rebuild.
  {
    auto discarded = first.AcquireFrame();
    ASSERT_TRUE(discarded);
  }
  if (GetParam() == BACKEND_API_VULKAN) {
    EXPECT_FALSE(first.AcquireFrame());
    EXPECT_FALSE(first.AcquireFrame());
  }
  ASSERT_TRUE(first.RebuildSwapChain());
  SubmissionToken last;
  for (int i = 0; i < 12; ++i) {
    ASSERT_TRUE(Window::PollEvents());
    if (i == 4 || i == 8) {
      ASSERT_TRUE(first.SetPresentMode(i == 4 ? PRESENT_MODE_CONTINUOUS : PRESENT_MODE_VSYNC));
      if (i == 8)
        EXPECT_EQ(first.GetPresentMode(), PRESENT_MODE_VSYNC);
    }
    for (auto &window : {first, second}) {
      auto frame = window.AcquireFrame();
      ASSERT_TRUE(frame);
      EXPECT_GT(frame.GetDesc().width, 0u);
      const auto supported = window.GetSupportedPresentationOptions();
      EXPECT_TRUE(std::any_of(supported.begin(), supported.end(), [&](const auto &option) {
        return option.format == frame.GetDesc().format && option.present_mode == window.GetPresentMode();
      }));
      auto context = device.CreateCommandContext(graphics);
      ASSERT_TRUE(context.CmdClearColor(frame, {0.6f, 0.7f, 0.8f, 1.0f}));
      ASSERT_TRUE(context.CmdPresent(window));
      frame = {};  // Recording already owns the exact frame.
      last = graphics.Submit(std::move(context));
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
  ASSERT_TRUE(old_context.CmdClearColor(old_frame, {0, 0, 0, 1}));
  ASSERT_TRUE(old_context.CmdPresent(first));
  auto old_packet = std::move(old_context);
  ASSERT_TRUE(MoveWindow(hwnd, 40, 40, 480, 360, TRUE));
  ASSERT_TRUE(Window::PollEvents());
  EXPECT_FALSE(first.AcquireFrame());
  EXPECT_TRUE(old_frame);  // Acquisition did not replace the swapchain.
  ASSERT_TRUE(first.RebuildSwapChain());
  EXPECT_FALSE(old_frame);
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
  ASSERT_TRUE(first.SetPresentMode(PRESENT_MODE_CONTINUOUS));
  glfwSetWindowUserPointer(first.GLFWWindow(), first.Get());
  glfwSetFramebufferSizeCallback(first.GLFWWindow(), [](GLFWwindow *native, int, int) {
    auto *window = static_cast<rhi::IWindow *>(glfwGetWindowUserPointer(native));
    EXPECT_TRUE(window->RebuildSwapChain());
  });
  ShowWindow(hwnd, SW_RESTORE);
  ASSERT_TRUE(Window::PollEvents());
  {
    auto restored = first.AcquireFrame();
    ASSERT_TRUE(restored);
  }
#endif
  auto upload = device.CreateCommandContext(transfer);
  auto ready = transfer.Submit(std::move(upload));
  ASSERT_TRUE(ready);
  auto frame = second.AcquireFrame();
  ASSERT_TRUE(frame);
  auto context = device.CreateCommandContext(graphics);
  ASSERT_TRUE(context.CmdClearColor(frame, {0, 0, 1, 1}));
  ASSERT_TRUE(context.CmdPresent(second));
  auto &window_backend = (*static_cast<backend::Window *>((second).Get()));
  frame.Release();
  first.Release();
  second.Release();
  // Only Device owns the Window. The recorded Image holds a frame lease,
  // not a strong Window reference, and keeps that lease until GPU completion.
  EXPECT_TRUE(window_backend.IsUniquelyOwned());
  EXPECT_TRUE(window_backend.HasFrameLeases());
  last = graphics.Submit(std::move(context), {{ready}});
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
