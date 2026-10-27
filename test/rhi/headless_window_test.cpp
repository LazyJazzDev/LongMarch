#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <future>

#include "grassland/rhi/rhix/rhix.h"

namespace grassland::rhi {
class HeadlessWindowTest : public testing::TestWithParam<BackendAPI> {
 protected:
  void SetUp() override {
    core = CreateCore(GetParam(), {true});
    ASSERT_TRUE(core);
    device = core->CreateDeviceByAutoSelection();
    ASSERT_TRUE(device);
  }

  ObjPtr<ICore> core;
  ObjPtr<IDevice> device;
};

TEST_P(HeadlessWindowTest, NoNativeWindowOrCallbacksAndFormatsFollowColorSpace) {
  ObjPtr<IWindow> window = device->CreateHeadlessWindow({16, 8, "headless"});
  ASSERT_TRUE(window);
  EXPECT_EQ(window->GLFWWindow(), nullptr);
  EXPECT_FALSE(window->ShouldClose());
  EXPECT_FALSE(window->IsKeyDown(GLFW_KEY_A));
  EXPECT_FALSE(window->IsSwapChainInvalid());
  bool callback = false;
  window->KeyEvent().RegisterCallback([&](int, int, int, int) { callback = true; });
  window->ResizeEvent().RegisterCallback([&](int, int) { callback = true; });
  window->FramebufferResizeEvent().RegisterCallback([&](int, int) { callback = true; });
  EXPECT_TRUE(window->EnableImGui());
  EXPECT_TRUE(window->UpdateImGui([&] { callback = true; }));
  EXPECT_TRUE(window->SetTitle("ignored"));
  EXPECT_TRUE(window->RebuildSwapChain());
  EXPECT_TRUE(window->SetPresentMode(PRESENT_MODE_VSYNC));
  EXPECT_EQ(window->GetPresentMode(), PRESENT_MODE_CONTINUOUS);
  float nits = 123;
  EXPECT_FALSE(window->GetHDRReferenceWhiteNits(nits));
  EXPECT_EQ(nits, 123);
  EXPECT_EQ(window->GetHDRReferenceWhiteScale(), 1);
  const std::array<std::pair<ColorSpace, ImageFormat>, 4> formats{
      {{COLOR_SPACE_SRGB_NONLINEAR, IMAGE_FORMAT_R8G8B8A8_UNORM},
       {COLOR_SPACE_EXTENDED_SRGB_LINEAR, IMAGE_FORMAT_R16G16B16A16_SFLOAT},
       {COLOR_SPACE_BT709_LINEAR, IMAGE_FORMAT_R16G16B16A16_SFLOAT},
       {COLOR_SPACE_HDR10_ST2084, IMAGE_FORMAT_R10G10B10A2_UNORM}}};
  for (const auto &[space, format] : formats) {
    ASSERT_TRUE(window->SetColorSpace(space));
    ObjPtr<IImage> image = window->GetNextImage();
    ASSERT_TRUE(image);
    EXPECT_EQ(image->GetDesc().format, format);
    EXPECT_EQ(image->GetDesc().width, 16);
    EXPECT_EQ(image->GetDesc().height, 8);
    ObjPtr<IImage> again = window->GetNextImage();
    EXPECT_EQ(image.Get(), again.Get());
    EXPECT_EQ(window->GetColorSpace(), space);
  }
  EXPECT_FALSE(callback);
  EXPECT_FALSE(window->SetColorSpace(static_cast<ColorSpace>(999)));
  EXPECT_FALSE(ObjPtr<IWindow>(device->CreateHeadlessWindow({0, 8})));
  EXPECT_FALSE(ObjPtr<IWindow>(device->CreateHeadlessWindow({8, -1})));
}

TEST_P(HeadlessWindowTest, ThreeLeasesIncludeRecordingsAndRetireAfterGPUCompletion) {
  ObjPtr<IWindow> window = device->CreateHeadlessWindow({16, 8});
  ASSERT_TRUE(window);
  std::array<ObjPtr<ICommandContext>, 3> commands;
  for (auto &cmd : commands) {
    ObjPtr<IImage> image = window->GetNextImage();
    ASSERT_TRUE(image);
    cmd = device->CreateCommandContext();
    ASSERT_TRUE(cmd);
    ASSERT_TRUE(cmd->CmdClearImage(image.Get(), {0.25f, 0.5f, 0.75f, 1.0f}));
    ASSERT_TRUE(cmd->CmdPresent(window.Get()));
  }
  EXPECT_FALSE(window->SetColorSpace(COLOR_SPACE_HDR10_ST2084));
  std::promise<void> started;
  auto entered = started.get_future();
  auto waiting = std::async(std::launch::async, [&] {
    started.set_value();
    return ObjPtr<IImage>(window->GetNextImage());
  });
  entered.wait();
  const auto before_release = waiting.wait_for(std::chrono::milliseconds(30));
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  ObjPtr<ISubmissionToken> done = queue->Submit(commands[0].Get());
  commands[0].Release();
  // Always release the other recordings on failure, so a failed assertion cannot
  // leave the waiting future blocked during stack unwinding.
  if (!done) {
    commands[1].Release();
    commands[2].Release();
  }
  ObjPtr<IImage> fourth = waiting.get();
  EXPECT_EQ(before_release, std::future_status::timeout);
  ASSERT_TRUE(done);
  EXPECT_TRUE(done->WaitHost());
  EXPECT_TRUE(fourth);
  commands[1].Release();
  commands[2].Release();
  fourth.Release();
  EXPECT_TRUE(window->SetColorSpace(COLOR_SPACE_HDR10_ST2084));
}

TEST_P(HeadlessWindowTest, DistinctLeasesAliasOneImageAndKeepItAliveWithoutWindow) {
  ObjPtr<IWindow> window = device->CreateHeadlessWindow({4, 4});
  ASSERT_TRUE(window);
  ObjPtr<IImage> first = window->GetNextImage();
  ObjPtr<ICommandContext> advance = device->CreateCommandContext();
  ASSERT_TRUE(advance->CmdPresent(window.Get()));
  advance.Release();
  ObjPtr<IImage> second = window->GetNextImage();
  ASSERT_TRUE(second);
  EXPECT_NE(first.Get(), second.Get());
  window.Release();
  ObjPtr<ICommandContext> cmd = device->CreateCommandContext();
  ASSERT_TRUE(cmd->CmdBeginRendering(
      {{{first.Get(), ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, {1.0f, 0.0f, 0.0f, 1.0f}}}}));
  ASSERT_TRUE(cmd->CmdEndRendering());
  ObjPtr<IDownloadToken> download = cmd->CmdDownloadImage(second.Get());
  ASSERT_TRUE(download);
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  ObjPtr<ISubmissionToken> done = queue->Submit(cmd.Get());
  ASSERT_TRUE(done);
  ASSERT_TRUE(done->WaitHost());
  std::array<uint8_t, 64> pixels{};
  ASSERT_TRUE(download->CopyTo(pixels.data(), pixels.size()));
  for (size_t i = 0; i < pixels.size(); i += 4) {
    EXPECT_EQ(pixels[i], 255);
    EXPECT_EQ(pixels[i + 1], 0);
    EXPECT_EQ(pixels[i + 2], 0);
    EXPECT_EQ(pixels[i + 3], 255);
  }
}

TEST_P(HeadlessWindowTest, RHIXPresentsAndReadsBackWithoutNativeWindow) {
  // Borrowed native device is independently retained by the RHIX handle.
  device->AddRef();
  rhix::Device wrapped = device.Get();
  auto window = wrapped.CreateHeadlessWindow({8, 8});
  ASSERT_TRUE(window);
  EXPECT_EQ(window.GLFWWindow(), nullptr);
  auto source = wrapped.CreateImage({8, 8, IMAGE_FORMAT_R8G8B8A8_UNORM});
  ASSERT_TRUE(source);
  auto queue = wrapped.GetQueue(QUEUE_TYPE_GRAPHICS);
  for (int frame = 0; frame < 12; ++frame) {
    auto image = window.GetNextImage();
    ASSERT_TRUE(image);
    auto cmd = wrapped.CreateCommandContext();
    ASSERT_TRUE(cmd.CmdClearImage(source, {0.0f, 1.0f, 0.0f, 1.0f}));
    ASSERT_TRUE(cmd.CmdPresentImage(window, source, COLOR_SPACE_SRGB_NONLINEAR));
    auto download = cmd.CmdDownloadImage(image);
    ASSERT_TRUE(download);
    auto done = queue.Submit(std::move(cmd));
    ASSERT_TRUE(done);
    ASSERT_TRUE(done.WaitHost());
    std::array<uint8_t, 256> pixels{};
    ASSERT_TRUE(download.CopyTo(pixels.data(), pixels.size()));
    EXPECT_EQ(pixels[0], 0);
    EXPECT_EQ(pixels[1], 255);
    EXPECT_EQ(pixels[2], 0);
    EXPECT_EQ(pixels[3], 255);
  }
}

TEST_P(HeadlessWindowTest, QueuedFramesKeepReadbackSnapshotsWithoutPerFrameHostWaits) {
  ObjPtr<IWindow> window = device->CreateHeadlessWindow({8, 8});
  ASSERT_TRUE(window);
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  std::vector<ObjPtr<ISubmissionToken>> submissions;
  std::vector<ObjPtr<IDownloadToken>> downloads;
  for (int frame = 0; frame < 32; ++frame) {
    ObjPtr<IImage> image = window->GetNextImage();
    ASSERT_TRUE(image);
    ObjPtr<ICommandContext> cmd = device->CreateCommandContext();
    ASSERT_TRUE(cmd->CmdClearImage(image.Get(), {frame / 255.0f, 0.0f, 0.0f, 1.0f}));
    ASSERT_TRUE(cmd->CmdPresent(window.Get()));
    downloads.emplace_back(cmd->CmdDownloadImage(image.Get()));
    ASSERT_TRUE(downloads.back());
    submissions.emplace_back(queue->Submit(cmd.Get()));
    ASSERT_TRUE(submissions.back());
    // Both local references expire here; only the GPU submission owns this lease.
  }
  for (int frame = 0; frame < 32; ++frame) {
    ASSERT_TRUE(submissions[frame]->WaitHost());
    std::array<uint8_t, 256> pixels{};
    ASSERT_TRUE(downloads[frame]->CopyTo(pixels.data(), pixels.size()));
    for (size_t offset = 0; offset < pixels.size(); offset += 4) {
      EXPECT_EQ(pixels[offset], frame);
      EXPECT_EQ(pixels[offset + 1], 0);
      EXPECT_EQ(pixels[offset + 2], 0);
      EXPECT_EQ(pixels[offset + 3], 255);
    }
  }
  EXPECT_TRUE(window->SetColorSpace(COLOR_SPACE_BT709_LINEAR));
}

TEST_P(HeadlessWindowTest, PresentRejectsDuplicateAndActiveRenderingWithoutLeakingLeases) {
  ObjPtr<IWindow> window = device->CreateHeadlessWindow({8, 8});
  ASSERT_TRUE(window);
  for (int frame = 0; frame < 8; ++frame) {
    ObjPtr<IImage> image = window->GetNextImage();
    ObjPtr<ICommandContext> cmd = device->CreateCommandContext();
    ASSERT_TRUE(cmd->CmdPresent(window.Get()));
    EXPECT_FALSE(cmd->CmdPresent(window.Get()));
    ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
    EXPECT_FALSE(ObjPtr<ISubmissionToken>(queue->Submit(cmd.Get())));
  }
  ObjPtr<IImage> image = window->GetNextImage();
  ObjPtr<ICommandContext> cmd = device->CreateCommandContext();
  ASSERT_TRUE(cmd->CmdBeginRendering(
      {{{image.Get(), ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, {0.0f, 0.0f, 0.0f, 1.0f}}}}));
  EXPECT_FALSE(cmd->CmdPresent(window.Get()));
  EXPECT_FALSE(cmd->CmdEndRendering());
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  EXPECT_FALSE(ObjPtr<ISubmissionToken>(queue->Submit(cmd.Get())));
}

const BackendAPI headless_backends[] = {
#if defined(LONGMARCH_D3D12_ENABLED)
    BACKEND_API_D3D12,
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
    BACKEND_API_VULKAN,
#endif
#if defined(LONGMARCH_METAL_ENABLED)
    BACKEND_API_METAL,
#endif
};
INSTANTIATE_TEST_SUITE_P(Backends, HeadlessWindowTest, testing::ValuesIn(headless_backends));
}  // namespace grassland::rhi
