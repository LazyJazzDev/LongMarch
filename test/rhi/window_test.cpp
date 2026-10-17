#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>

#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#endif

#include "grassland/rhi/backend/backend_window.h"
#include "grassland/rhi/rhix/rhix.h"

#if defined(LONGMARCH_D3D12_ENABLED)
#include <d3d12sdklayers.h>

#include "grassland/rhi/backend/d3d12/d3d12_device.h"
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
  auto graphics = device.GetQueue(QUEUE_TYPE_GRAPHICS);
  auto transfer = device.GetQueue(QUEUE_TYPE_TRANSFER);
  ASSERT_TRUE(graphics);
  ASSERT_TRUE(transfer);
  EXPECT_TRUE(graphics.SharesPhysicalQueue(transfer));
  auto first = device.CreateGLFWWindow({320, 240, "RHI window test A"});
  auto second = device.CreateGLFWWindow({320, 240, "RHI window test B"});
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  int user_data = 42;
  glfwSetWindowUserPointer(first.GLFWWindow(), &user_data);
#if !defined(NDEBUG)
  {
    auto frame = first.GetNextImage();
    ASSERT_TRUE(frame);
    auto context = device.CreateCommandContext();
    ASSERT_TRUE(context.CmdClearColor(frame, {0, 0, 0, 1}));
    ASSERT_TRUE(context.CmdPresent(first));
    EXPECT_FALSE(context.CmdPresent(first));
    EXPECT_FALSE(graphics.Submit(std::move(context)));
  }
#endif
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
  // Dropping a caller reference does not advance or invalidate the next image.
  auto next = first.GetNextImage();
  ASSERT_TRUE(next);
  auto same = first.GetNextImage();
  EXPECT_EQ(next.Get(), same.Get());
  same.Release();
  same = first.GetNextImage();
  EXPECT_EQ(next.Get(), same.Get());
  same.Release();
  next.Release();
  SubmissionToken last;
  for (int i = 0; i < 12; ++i) {
    ASSERT_TRUE(Window::PollEvents());
    if (i == 4 || i == 8) {
      ASSERT_TRUE(first.SetPresentMode(i == 4 ? PRESENT_MODE_CONTINUOUS : PRESENT_MODE_VSYNC));
      if (i == 8)
        EXPECT_EQ(first.GetPresentMode(), PRESENT_MODE_VSYNC);
    }
    for (auto &window : {first, second}) {
      auto frame = window.GetNextImage();
      ASSERT_TRUE(frame);
      EXPECT_GT(frame.GetDesc().width, 0u);
      const auto supported = window.GetSupportedPresentationOptions();
      EXPECT_TRUE(std::any_of(supported.begin(), supported.end(), [&](const auto &option) {
        return option.format == frame.GetDesc().format && option.present_mode == window.GetPresentMode();
      }));
      auto context = device.CreateCommandContext();
      ASSERT_TRUE(context.CmdClearColor(frame, {1.0f, 0.0f, 0.0f, 1.0f}));
      ASSERT_TRUE(context.CmdClearColor(frame, {0.6f, 0.7f, 0.8f, 1.0f}));
      ASSERT_TRUE(context.CmdPresent(window));
      auto before_submit = window.GetNextImage();
      EXPECT_EQ(frame.Get(), before_submit.Get());
      frame = {};  // Recording already owns the exact frame.
      if (window.Get() == first.Get())
        last = transfer.Submit(std::move(context));
      else
        last = graphics.Submit(std::move(context));
      ASSERT_TRUE(last);
      auto after_submit = window.GetNextImage();
      ASSERT_TRUE(after_submit);
      EXPECT_NE(before_submit.Get(), after_submit.Get());
      EXPECT_EQ(after_submit.Get(), window.GetNextImage().Get());
    }
  }
  ASSERT_TRUE(last.WaitHost());
#ifdef _WIN32
  const auto hwnd = FindWindowA(nullptr, "RHI window test A");
  ASSERT_NE(hwnd, nullptr);
  auto old_frame = first.GetNextImage();
  ASSERT_TRUE(old_frame);
  auto old_context = device.CreateCommandContext();
  ASSERT_TRUE(old_context.CmdClearColor(old_frame, {0, 0, 0, 1}));
  ASSERT_TRUE(old_context.CmdPresent(first));
  old_context.Release();
  old_frame.Release();
  ASSERT_TRUE(MoveWindow(hwnd, 40, 40, 480, 360, TRUE));
  ASSERT_TRUE(Window::PollEvents());
  EXPECT_EQ(glfwGetWindowUserPointer(first.GLFWWindow()), &user_data);
  {
    auto resized = first.GetNextImage();
    ASSERT_TRUE(resized);
  }
  ShowWindow(hwnd, SW_MINIMIZE);
  ASSERT_TRUE(Window::PollEvents());
  EXPECT_FALSE(first.GetNextImage());
  ASSERT_TRUE(first.SetPresentMode(PRESENT_MODE_CONTINUOUS));
  ShowWindow(hwnd, SW_RESTORE);
  ASSERT_TRUE(Window::PollEvents());
  {
    auto restored = first.GetNextImage();
    ASSERT_TRUE(restored);
  }
#endif
  auto upload = device.CreateCommandContext();
  auto ready = transfer.Submit(std::move(upload));
  ASSERT_TRUE(ready);
  auto frame = second.GetNextImage();
  ASSERT_TRUE(frame);
  auto context = device.CreateCommandContext();
  ASSERT_TRUE(context.CmdClearColor(frame, {0, 0, 1, 1}));
  ASSERT_TRUE(context.CmdPresent(second));
  frame.Release();
  first.Release();
#ifdef _WIN32
  EXPECT_FALSE(IsWindow(hwnd));
  const auto second_hwnd = FindWindowA(nullptr, "RHI window test B");
  ASSERT_NE(second_hwnd, nullptr);
#endif
  last = graphics.Submit(std::move(context), {{ready}});
  ASSERT_TRUE(last);
  second.Release();  // Destruction waits for native GPU use; Device does not retain Window.
#ifdef _WIN32
  EXPECT_FALSE(IsWindow(second_hwnd));
#endif
  ASSERT_TRUE(last.WaitHost());
  last.Release();
  ready.Release();
  graphics.Release();
  transfer.Release();
  device.Release();  // Device drains work after caller-owned children are released.
  core.Release();
  EXPECT_TRUE(Window::PollEvents());
}

TEST_P(RhiWindow, MultipleImagesRestoreDefaultsAcrossContextsAndPresentWithoutClear) {
  if (!std::getenv("LONGMARCH_RHI_WINDOW_TESTS"))
    GTEST_SKIP() << "Set LONGMARCH_RHI_WINDOW_TESTS=1 to run native resource-state tests.";
#if defined(LONGMARCH_D3D12_ENABLED)
  backend::d3d12::ComPtr<ID3D12Debug> debug;
  if (GetParam() == BACKEND_API_D3D12 && std::getenv("LONGMARCH_RHI_VALIDATION")) {
    ASSERT_TRUE(SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug))));
    debug->EnableDebugLayer();
  }
#endif
  auto core = rhix::CreateCore(GetParam());
  ASSERT_TRUE(core);
  auto device = core.CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
#if defined(LONGMARCH_D3D12_ENABLED)
  backend::d3d12::ComPtr<ID3D12InfoQueue> messages;
  if (debug) {
    auto native = static_cast<backend::d3d12::D3D12Device *>(device.Get());
    ASSERT_TRUE(SUCCEEDED(native->NativeDevice()->QueryInterface(IID_PPV_ARGS(&messages))));
    messages->ClearStoredMessages();
  }
#endif
  auto queue = device.GetQueue(QUEUE_TYPE_GRAPHICS);
  auto first = device.CreateGLFWWindow({320, 240, "RHI state test A"});
  auto second = device.CreateGLFWWindow({320, 240, "RHI state test B"});
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  auto a = first.GetNextImage();
  auto b = second.GetNextImage();
  ASSERT_TRUE(a);
  ASSERT_TRUE(b);
  {
    auto abandoned = device.CreateCommandContext();
    ASSERT_TRUE(abandoned.CmdClearColor(a, {1, 0, 0, 1}));
    ASSERT_TRUE(abandoned.CmdClearColor(b, {0, 1, 0, 1}));
  }  // Recording alone must not change the next context's initial resource state.
  auto clear = device.CreateCommandContext();
  ASSERT_TRUE(clear.CmdClearColor(a, {1, 0, 0, 1}));
  ASSERT_TRUE(clear.CmdClearColor(b, {0, 1, 0, 1}));
  ASSERT_TRUE(clear.CmdClearColor(a, {0, 0, 1, 1}));
  auto cleared = queue.Submit(std::move(clear));
  ASSERT_TRUE(cleared);
  ASSERT_TRUE(cleared.WaitHost());
  EXPECT_EQ(a.Get(), first.GetNextImage().Get());
  EXPECT_EQ(b.Get(), second.GetNextImage().Get());
  auto reclear = device.CreateCommandContext();
  ASSERT_TRUE(reclear.CmdClearColor(b, {0.6f, 0.7f, 0.8f, 1}));
  ASSERT_TRUE(reclear.CmdClearColor(a, {0.6f, 0.7f, 0.8f, 1}));
  auto ready = queue.Submit(std::move(reclear), {{cleared}});
  ASSERT_TRUE(ready);
  auto present = device.CreateCommandContext();
  ASSERT_TRUE(present.CmdPresent(second));
  ASSERT_TRUE(present.CmdPresent(first));
  auto done = queue.Submit(std::move(present), {{ready}});
  ASSERT_TRUE(done);
  ASSERT_TRUE(done.WaitHost());
  EXPECT_NE(a.Get(), first.GetNextImage().Get());
  EXPECT_NE(b.Get(), second.GetNextImage().Get());
  a.Release();
  b.Release();
  // Present without a clear must also transition a fresh image into its presentable layout.
  auto next = device.CreateCommandContext();
  ASSERT_TRUE(next.CmdPresent(first));
  auto next_done = queue.Submit(std::move(next));
  ASSERT_TRUE(next_done);
  EXPECT_TRUE(next_done.WaitHost());
  auto cleanup_image = second.GetNextImage();
  auto cleanup = device.CreateCommandContext();
  ASSERT_TRUE(cleanup.CmdClearColor(cleanup_image, {0, 0, 0, 1}));
  auto cleanup_done = queue.Submit(std::move(cleanup));
  ASSERT_TRUE(cleanup_done);
  cleanup_image.Release();
  second.Release();  // A pending clear without Present must not require Window in image destruction.
  EXPECT_TRUE(cleanup_done.WaitHost());
  // Exercise reuse across abandoned recordings, asynchronous batches and retained handles.
  // Retained contexts must not return their native resources while still caller-owned.
  for (int batch = 0; batch < 8; ++batch) {
    ASSERT_TRUE(first.SetPresentMode(batch % 2 ? PRESENT_MODE_VSYNC : PRESENT_MODE_CONTINUOUS));
    std::vector<rhi::ObjPtr<rhi::ICommandContext>> held;
    rhi::ObjPtr<rhi::ISubmissionToken> completed;
    for (int frame = 0; frame < 8; ++frame) {
      rhi::ObjPtr<rhi::IImage> image = first.Get()->GetNextImage();
      ASSERT_TRUE(image);
      {
        rhi::ObjPtr<rhi::ICommandContext> abandoned = device.Get()->CreateCommandContext();
        ASSERT_TRUE(abandoned);
        ASSERT_TRUE(abandoned->CmdClearColor(image.Get(), {1, 0, 0, 1}));
      }
      rhi::ObjPtr<rhi::ICommandContext> recording = device.Get()->CreateCommandContext();
      ASSERT_TRUE(recording);
      ASSERT_TRUE(recording->CmdClearColor(image.Get(), {0.6f, 0.7f, 0.8f, 1}));
      ASSERT_TRUE(recording->CmdPresent(first.Get()));
      completed = queue.Get()->Submit(recording.Get());
      ASSERT_TRUE(completed);
      held.push_back(std::move(recording));
    }
    ASSERT_TRUE(completed->WaitHost());
    // The next batch resets resources only after both completion and handle release.
    held.clear();
  }
#if defined(LONGMARCH_D3D12_ENABLED)
  if (messages) {
    for (UINT64 i = 0; i < messages->GetNumStoredMessages(); ++i) {
      SIZE_T size{};
      ASSERT_TRUE(SUCCEEDED(messages->GetMessage(i, nullptr, &size)));
      std::vector<uint8_t> storage(size);
      auto message = reinterpret_cast<D3D12_MESSAGE *>(storage.data());
      ASSERT_TRUE(SUCCEEDED(messages->GetMessage(i, message, &size)));
      EXPECT_NE(message->Severity, D3D12_MESSAGE_SEVERITY_ERROR) << message->pDescription;
      EXPECT_NE(message->Severity, D3D12_MESSAGE_SEVERITY_CORRUPTION) << message->pDescription;
    }
  }
#endif
}

TEST_P(RhiWindow, ExplicitRebuildPreservesModeAndRecoversFromMinimization) {
  if (!std::getenv("LONGMARCH_RHI_WINDOW_TESTS"))
    GTEST_SKIP() << "Requires a desktop.";
  auto core = rhix::CreateCore(GetParam());
  ASSERT_TRUE(core);
  auto device = core.CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  auto window = device.CreateGLFWWindow({320, 240, "RHI explicit rebuild"});
  ASSERT_TRUE(window);
  EXPECT_FALSE(window.IsSwapChainInvalid());
  ASSERT_TRUE(window.SetPresentMode(PRESENT_MODE_CONTINUOUS));
  const auto mode = window.GetPresentMode();
  ASSERT_TRUE(window.Get()->RebuildSwapChain());
  EXPECT_FALSE(window.Get()->IsSwapChainInvalid());
  EXPECT_EQ(window.GetPresentMode(), mode);
  {
    auto image = window.GetNextImage();
    ASSERT_TRUE(image);
    EXPECT_EQ(image.Get(), window.GetNextImage().Get());
  }
#ifdef _WIN32
  const auto hwnd = FindWindowA(nullptr, "RHI explicit rebuild");
  ASSERT_NE(hwnd, nullptr);
  ShowWindow(hwnd, SW_MINIMIZE);
  ASSERT_TRUE(Window::PollEvents());
  ASSERT_TRUE(window.RebuildSwapChain());
  EXPECT_TRUE(window.IsSwapChainInvalid());
  EXPECT_FALSE(window.GetNextImage());
  ShowWindow(hwnd, SW_RESTORE);
  ASSERT_TRUE(Window::PollEvents());
  EXPECT_FALSE(window.IsSwapChainInvalid());
  EXPECT_EQ(window.GetPresentMode(), mode);
  EXPECT_TRUE(window.GetNextImage());
#endif
}

#if defined(_WIN32) && defined(LONGMARCH_VULKAN_ENABLED)
TEST(RhiVulkanWindow, OutOfDateDefersRebuildAndKeepsSubmittedWorkAlive) {
  if (!std::getenv("LONGMARCH_RHI_WINDOW_TESTS"))
    GTEST_SKIP() << "Requires a desktop.";
  auto core = rhix::CreateCore(BACKEND_API_VULKAN);
  ASSERT_TRUE(core);
  auto device = core.CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  auto window = device.CreateGLFWWindow({320, 240, "RHI out of date"});
  ASSERT_TRUE(window);
  auto queue = device.GetQueue(QUEUE_TYPE_GRAPHICS);
  ASSERT_TRUE(queue);
  ASSERT_TRUE(Window::PollEvents());
  auto native = window.GLFWWindow();
  // Suppress event-driven rebuilding to reproduce a surface change observed
  // by WSI before the application processes its resize notification.
  const auto callback = glfwSetFramebufferSizeCallback(native, nullptr);
  bool invalid = false;
  for (int i = 0; i < 8 && !invalid; ++i) {
    auto image = window.GetNextImage();
    ASSERT_TRUE(image);
    auto context = device.CreateCommandContext();
    ASSERT_TRUE(context.CmdClearColor(image, {0.6f, 0.7f, 0.8f, 1}));
    ASSERT_TRUE(context.CmdPresent(window));
    glfwSetWindowSize(native, 400 + 40 * i, 300 + 30 * i);
    // No callback has replaced the prepared object, even though the native
    // surface changed. Submit may observe OUT_OF_DATE in Present or Acquire.
    EXPECT_EQ(image.Get(), window.GetNextImage().Get());
    auto token = queue.Submit(std::move(context));
    ASSERT_TRUE(token);
    ASSERT_TRUE(token.WaitHost());
    invalid = window.IsSwapChainInvalid();
    if (invalid) {
      EXPECT_FALSE(window.GetNextImage());
      EXPECT_TRUE(window.IsSwapChainInvalid());
      EXPECT_EQ(image.GetDesc().width, 320u);
    }
  }
  glfwSetFramebufferSizeCallback(native, callback);
  if (!invalid)
    GTEST_SKIP() << "Driver permits resized presentation without OUT_OF_DATE.";
  // All caller-held frames and recordings are gone before explicit rebuild.
  ASSERT_TRUE(window.RebuildSwapChain());
  EXPECT_FALSE(window.IsSwapChainInvalid());
  for (int i = 0; i < 24; ++i) {
    auto image = window.GetNextImage();
    ASSERT_TRUE(image);
    int width, height;
    glfwGetFramebufferSize(native, &width, &height);
    EXPECT_EQ(image.GetDesc().width, static_cast<uint32_t>(width));
    EXPECT_EQ(image.GetDesc().height, static_cast<uint32_t>(height));
    auto context = device.CreateCommandContext();
    ASSERT_TRUE(context.CmdClearColor(image, {0, 0, 1, 1}));
    ASSERT_TRUE(context.CmdPresent(window));
    auto token = queue.Submit(std::move(context));
    ASSERT_TRUE(token);
    ASSERT_TRUE(token.WaitHost());
  }
}
#endif

std::vector<BackendAPI> WindowBackends() {
  std::vector<BackendAPI> backends;
#ifdef LONGMARCH_D3D12_ENABLED
  backends.push_back(BACKEND_API_D3D12);
#endif
#ifdef LONGMARCH_VULKAN_ENABLED
  backends.push_back(BACKEND_API_VULKAN);
#endif
#ifdef LONGMARCH_METAL_ENABLED
  backends.push_back(BACKEND_API_METAL);
#endif
  return backends;
}

INSTANTIATE_TEST_SUITE_P(Backends, RhiWindow, testing::ValuesIn(WindowBackends()));

}  // namespace
}  // namespace grassland::rhix
