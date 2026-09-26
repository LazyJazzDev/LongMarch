#include <gtest/gtest.h>
#include <long_march.h>

#include <cmath>
#include <cstdlib>
#include <glm/gtc/packing.hpp>

#if defined(LONGMARCH_D3D12_ENABLED)
#include "grassland/graphics/backend/d3d12/d3d12_window.h"
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
#include "grassland/graphics/backend/vulkan/vulkan_window.h"
#endif

using namespace grassland;

namespace {
bool InteractiveHDRTests() {
  return std::getenv("LONGMARCH_TEST_HDR") || std::getenv("LONGMARCH_TEST_HDR_WINDOWS");
}

bool HasHDRSurface(graphics::Window *window) {
#if defined(LONGMARCH_VULKAN_ENABLED)
  if (auto native = dynamic_cast<graphics::backend::VulkanWindow *>(window)) {
    auto swapchain = native->SwapChain();
    const auto support = vulkan::Swapchain::QuerySwapChainSupport(swapchain->Device()->PhysicalDevice().Handle(),
                                                                  swapchain->Surface()->Handle());
    return graphics::backend::VulkanWindow::ChooseHDRSurfaceFormat(support.formats).has_value();
  }
#endif
  return true;
}

double PQ(double nits) {
  const double p = std::pow(std::clamp(nits / 10000.0, 0.0, 1.0), 2610.0 / 16384.0);
  return std::pow((3424.0 / 4096.0 + 2413.0 / 128.0 * p) / (1.0 + 2392.0 / 128.0 * p), 2523.0 / 32.0);
}

bool UsesPQSwapchain(graphics::Window *window) {
#if defined(LONGMARCH_VULKAN_ENABLED)
  if (auto *native = dynamic_cast<graphics::backend::VulkanWindow *>(window)) {
    // Inspect the actual backend target, without a public encoding query.
    auto format = native->SwapChain()->Format();
    return format == VK_FORMAT_A2B10G10R10_UNORM_PACK32 || format == VK_FORMAT_A2R10G10B10_UNORM_PACK32;
  }
#endif
  return false;
}

glm::vec3 ExpectedPQ(glm::vec3 linear) {
  const glm::dvec3 bt2020{0.627404 * linear.r + 0.329283 * linear.g + 0.043313 * linear.b,
                          0.069097 * linear.r + 0.919540 * linear.g + 0.011362 * linear.b,
                          0.016391 * linear.r + 0.088013 * linear.g + 0.895595 * linear.b};
  const auto nits = bt2020 * 203.0;
  return {PQ(nits.r), PQ(nits.g), PQ(nits.b)};
}

glm::vec3 ExpectedOutput(graphics::Window *window, glm::vec3 linear) {
  return UsesPQSwapchain(window) ? ExpectedPQ(linear) : linear * window->HDRReferenceWhiteScale();
}
}  // namespace

#if defined(LONGMARCH_VULKAN_ENABLED)
// Reproduce the GUI's Cornell Box (square) -> Texture (wide) resize on the
// same window. Do not toggle HDR between sizes: that would hide a missed resize
// notification by forcing an unrelated swapchain rebuild.
TEST(VulkanWindowResizeTest, ProgrammaticSquareToWideUpdatesPresentation) {
  if (!InteractiveHDRTests())
    GTEST_SKIP() << "Requires an interactive desktop session";
  std::unique_ptr<graphics::Core> core;
  ASSERT_EQ(graphics::CreateCore(graphics::BACKEND_API_VULKAN, graphics::Core::Settings{2, true}, &core), 0);
  ASSERT_EQ(core->InitializeLogicalDeviceAutoSelect(false), 0);
  for (bool hdr : {false, true}) {
    auto window = std::unique_ptr<graphics::Window>{};
    ASSERT_EQ(core->CreateWindowObject(1024, 1024, "Scene resize regression", &window), 0);
    if (hdr && !HasHDRSurface(window.get()))
      continue;  // X11 still exercises the complete SDR resize path.
    ASSERT_EQ(window->SetHDR(hdr), 0);
    window->InitImGui();
    ImGui::GetIO().IniFilename = nullptr;
    auto *native = dynamic_cast<graphics::backend::VulkanWindow *>(window.get());
    ASSERT_NE(native, nullptr);
    for (auto size : {glm::ivec2{1024, 1024}, glm::ivec2{2048, 1024}, glm::ivec2{1024, 1024}}) {
      window->Resize(size.x, size.y);
      // X11 acknowledges asynchronously. Present through several event cycles
      // to also cover Wayland fractional scaling and compositor configure events.
      for (int frame = 0; frame < 4; ++frame) {
        glfwWaitEventsTimeout(0.02);
        auto fb = window->GetFramebufferSize();
        auto extent = native->SwapChain()->Extent();
        SCOPED_TRACE(testing::Message() << "hdr=" << hdr << " requested=" << size.x << "x" << size.y
                                        << " framebuffer=" << fb.x << "x" << fb.y << " frame=" << frame);
        ASSERT_EQ(extent.width, fb.x);
        ASSERT_EQ(extent.height, fb.y);
        std::unique_ptr<graphics::Image> source;
        ASSERT_EQ(core->CreateImage(fb.x, fb.y, graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT, &source), 0);
        window->BeginImGuiFrame();
        ImGui::TextUnformatted("Scene resize regression");
        window->EndImGuiFrame();
        std::unique_ptr<graphics::CommandContext> commands;
        ASSERT_EQ(core->CreateCommandContext(&commands), 0);
        commands->CmdClearImage(source.get(), {{0.5f, 0.5f, 0.5f, 1.0f}});
        commands->CmdPresent(window.get(), source.get());
        ASSERT_EQ(core->SubmitCommandContext(commands.get()), 0);
        core->WaitGPU();
        if (hdr) {
          std::unique_ptr<graphics::CommandContext> readback;
          ASSERT_EQ(core->CreateCommandContext(&readback), 0);
          auto *aligned = window->AlignHDRComposition(readback.get());
          ASSERT_EQ(aligned->Extent().width, fb.x);
          ASSERT_EQ(aligned->Extent().height, fb.y);
          ASSERT_EQ(core->SubmitCommandContext(readback.get()), 0);
          core->WaitGPU();
          std::vector<uint16_t> pixels(size_t(fb.x) * fb.y * 4);
          aligned->DownloadData(pixels.data());
          // Check both bottom corners: a stale square target leaves a black
          // border, crops the image or fails to write the newly exposed area.
          for (int x : {0, fb.x - 1}) {
            size_t offset = (size_t(fb.y - 1) * fb.x + x) * 4;
            auto expected = ExpectedOutput(window.get(), glm::vec3{0.5f});
            for (int c = 0; c < 3; ++c)
              EXPECT_NEAR(glm::unpackHalf1x16(pixels[offset + c]), expected[c], 0.01f);
          }
        }
      }
    }
    window->CloseWindow();
  }
}

TEST(HDRSurfaceFormatTest, PreferredFormatWinsRegardlessOfEnumerationOrder) {
  const VkSurfaceFormatKHR hdr{VK_FORMAT_R16G16B16A16_SFLOAT, VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT};
  const VkSurfaceFormatKHR sdr{VK_FORMAT_R8G8B8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
  for (const auto &formats : {std::vector<VkSurfaceFormatKHR>{hdr, sdr}, std::vector<VkSurfaceFormatKHR>{sdr, hdr}}) {
    auto selected = vulkan::Swapchain::ChooseSwapSurfaceFormat(formats, hdr.format, hdr.colorSpace);
    EXPECT_EQ(selected.format, hdr.format);
    EXPECT_EQ(selected.colorSpace, hdr.colorSpace);
    selected = vulkan::Swapchain::ChooseSwapSurfaceFormat(formats, sdr.format, sdr.colorSpace);
    EXPECT_EQ(selected.format, sdr.format);
    EXPECT_EQ(selected.colorSpace, sdr.colorSpace);
  }
  EXPECT_THROW(vulkan::Swapchain::ChooseSwapSurfaceFormat({}, hdr.format, hdr.colorSpace), std::runtime_error);
}

TEST(HDRSurfaceFormatTest, HDRNegotiationPreservesScRGBAndSupportsBothPQLayouts) {
  using graphics::backend::VulkanWindow;
  const VkSurfaceFormatKHR scrgb{VK_FORMAT_R16G16B16A16_SFLOAT, VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT};
  const VkSurfaceFormatKHR pq{VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_COLOR_SPACE_HDR10_ST2084_EXT};
  const VkSurfaceFormatKHR pq_bgr{VK_FORMAT_A2R10G10B10_UNORM_PACK32, VK_COLOR_SPACE_HDR10_ST2084_EXT};
  const VkSurfaceFormatKHR sdr{VK_FORMAT_R8G8B8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
  EXPECT_EQ(VulkanWindow::ChooseHDRSurfaceFormat({pq, sdr, scrgb})->format, scrgb.format);
  EXPECT_EQ(VulkanWindow::ChooseHDRSurfaceFormat({sdr, pq_bgr, pq})->format, pq.format);
  EXPECT_EQ(VulkanWindow::ChooseHDRSurfaceFormat({sdr, pq_bgr})->format, pq_bgr.format);
  EXPECT_FALSE(VulkanWindow::ChooseHDRSurfaceFormat({sdr}));
  EXPECT_FALSE(VulkanWindow::ChooseHDRSurfaceFormat({}));
  // Float storage or 10-bit precision alone does not establish HDR encoding.
  EXPECT_FALSE(VulkanWindow::ChooseHDRSurfaceFormat(
      {{scrgb.format, VK_COLOR_SPACE_BT709_LINEAR_EXT}, {pq.format, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR}}));
}
#endif

class BrightnessProbeWindow : public graphics::Window {
 public:
  BrightnessProbeWindow() : Window(64, 64, "Brightness query test", false, false, false) {
  }

  graphics::DisplayBrightness brightness;
  bool pq_output{false};

  void InitImGui(const char *, float) override {
  }

  void TerminateImGui() override {
  }

  void BeginImGuiFrame() override {
  }

  void EndImGuiFrame() override {
  }

  ImGuiContext *GetImGuiContext() const override {
    return nullptr;
  }

 protected:
  bool UsesPQOutput() const override {
    return pq_output;
  }

  graphics::DisplayBrightness QueryDisplayBrightness() const override {
    return brightness;
  }
};

TEST(HDRBrightnessTest, SetHDRReportsFailureWithoutThrowing) {
  if (!InteractiveHDRTests())
    GTEST_SKIP();
  BrightnessProbeWindow window;
  window.brightness = {240.0f, 3.0f, 0.0f, true, true};
  ASSERT_EQ(window.SetHDR(true), 0);
  auto callback = window.ResizeEvent().RegisterCallback(
      [](int, int) { throw std::runtime_error("Injected presentation notification failure"); });
  EXPECT_NE(window.SetHDR(false), 0);
  EXPECT_FLOAT_EQ(window.HDRReferenceWhiteScale(), 3.0f);
  window.ResizeEvent().UnregisterCallback(callback);
  EXPECT_EQ(window.SetHDR(false), 0);
  EXPECT_FLOAT_EQ(window.HDRReferenceWhiteScale(), 1.0f);
  window.CloseWindow();
  EXPECT_NE(window.SetHDR(true), 0);
  EXPECT_NE(window.SetHDR(false), 0);
}

TEST(HDRBrightnessTest, RefreshNotifiesAndUnknownReferenceFallsBack) {
  if (!InteractiveHDRTests())
    GTEST_SKIP();
  BrightnessProbeWindow window;
  int changes = 0;
  window.DisplayBrightnessEvent().RegisterCallback([&](const graphics::DisplayBrightness &) { ++changes; });
  window.brightness = {480.0f, 6.0f, 0.0f, true, true};
  window.RefreshDisplayBrightness();
  EXPECT_EQ(changes, 1);
  EXPECT_EQ(window.HDRReferenceWhiteScale(), 1.0f);  // SDR must never scale.
  ASSERT_EQ(window.SetHDR(true), 0);
  EXPECT_EQ(window.HDRReferenceWhiteScale(), 6.0f);
  EXPECT_EQ(changes, 1);
  window.brightness = {160.0f, 2.0f, 0.0f, true, true};
  window.RefreshDisplayBrightness();
  EXPECT_EQ(changes, 2);
  EXPECT_EQ(window.HDRReferenceWhiteScale(), 2.0f);
  window.brightness = {};
  window.RefreshDisplayBrightness();
  EXPECT_EQ(changes, 3);
  EXPECT_EQ(window.HDRReferenceWhiteScale(), 1.0f);
  EXPECT_FALSE(window.GetDisplayBrightness().reference_white_known);
  // Headroom changes must notify even when reference white is unchanged.
  window.brightness.hdr_headroom = 4.0f;
  window.RefreshDisplayBrightness();
  EXPECT_EQ(changes, 4);
  window.brightness.hdr_headroom = 2.0f;
  window.RefreshDisplayBrightness();
  EXPECT_EQ(changes, 5);
  window.RefreshDisplayBrightness();
  EXPECT_EQ(changes, 5);
}

#if defined(LONGMARCH_D3D12_ENABLED) || defined(LONGMARCH_VULKAN_ENABLED)
class HDRWindowTest : public testing::TestWithParam<graphics::BackendAPI> {};

TEST_P(HDRWindowTest, PresentationAndImGuiSwitching) {
  if (!InteractiveHDRTests())
    GTEST_SKIP() << "Set LONGMARCH_TEST_HDR=1 in an interactive desktop session";
  std::unique_ptr<graphics::Core> core;
  ASSERT_EQ(graphics::CreateCore(GetParam(), graphics::Core::Settings{2, true}, &core), 0);
  ASSERT_EQ(core->InitializeLogicalDeviceAutoSelect(false), 0);
  for (bool imgui : {false, true}) {
    std::unique_ptr<graphics::Window> window;
    ASSERT_EQ(core->CreateWindowObject(320, 240, "HDR presentation test", &window), 0);
    if (!HasHDRSurface(window.get()))
      GTEST_SKIP() << "Desktop exposes no supported HDR surface (SDR fallback tested separately)";
    if (imgui) {
      window->InitImGui();
      ImGui::GetIO().IniFilename = nullptr;
    }
    std::unique_ptr<graphics::Image> image;
    ASSERT_EQ(core->CreateImage(320, 240, graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT, &image), 0);
    bool resized = false;
    for (bool hdr : {true, true, false, true, false}) {
      ASSERT_EQ(window->SetHDR(hdr), 0);
#if defined(LONGMARCH_D3D12_ENABLED)
      if (GetParam() == graphics::BACKEND_API_D3D12) {
        auto native = dynamic_cast<graphics::backend::D3D12Window *>(window.get());
        EXPECT_EQ(native->SwapChain()->BackBufferFormat(),
                  hdr ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_R8G8B8A8_UNORM);
      }
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
      if (GetParam() == graphics::BACKEND_API_VULKAN) {
        auto native = dynamic_cast<graphics::backend::VulkanWindow *>(window.get());
        auto *swapchain = native->SwapChain();
        const auto support = vulkan::Swapchain::QuerySwapChainSupport(swapchain->Device()->PhysicalDevice().Handle(),
                                                                      swapchain->Surface()->Handle());
        const auto expected_format =
            hdr ? graphics::backend::VulkanWindow::ChooseHDRSurfaceFormat(support.formats)->format
                : VK_FORMAT_R8G8B8A8_UNORM;
        EXPECT_EQ(swapchain->Format(), expected_format);
      }
#endif
      if (imgui) {
        ImGui::GetIO().IniFilename = nullptr;
        window->BeginImGuiFrame();
        ImGui::TextUnformatted("HDR / SDR switching");
        ImGui::GetForegroundDrawList()->AddRectFilled({100, 100}, {220, 200}, IM_COL32(128, 128, 128, 255));
        window->EndImGuiFrame();
      }
      std::unique_ptr<graphics::CommandContext> commands;
      ASSERT_EQ(core->CreateCommandContext(&commands), 0);
      commands->CmdClearImage(image.get(), {{3.0f, 2.0f, 1.0f, 1.0f}});
      commands->CmdPresent(window.get(), image.get());
      ASSERT_EQ(core->SubmitCommandContext(commands.get()), 0);
      core->WaitGPU();
      if (hdr) {
        std::unique_ptr<graphics::CommandContext> readback_commands;
        core->CreateCommandContext(&readback_commands);
        auto *aligned = window->AlignHDRComposition(readback_commands.get());
        ASSERT_EQ(core->SubmitCommandContext(readback_commands.get()), 0);
        core->WaitGPU();
        std::vector<uint16_t> pixels(aligned->Extent().width * aligned->Extent().height * 4);
        aligned->DownloadData(pixels.data());
        const float scale = window->HDRReferenceWhiteScale();
        const size_t center =
            ((aligned->Extent().height / 2) * aligned->Extent().width + aligned->Extent().width / 2) * 4;
        // ImGui's 128/255 gray is sRGB, decoded to about 0.216 linear.
        const auto expected = ExpectedOutput(window.get(), imgui ? glm::vec3{0.216f} : glm::vec3{3, 2, 1});
        for (int c = 0; c < 3; ++c)
          EXPECT_NEAR(glm::unpackHalf1x16(pixels[center + c]), expected[c], 0.01f * scale);
      }
      if (!resized) {
        window->Resize(352, 256);
        resized = true;
      }
      glfwPollEvents();
    }
    window->CloseWindow();
  }
  core->WaitGPU();
}

TEST_P(HDRWindowTest, ReferenceWhiteScalingPreservesSourceAndAlpha) {
  if (!InteractiveHDRTests())
    GTEST_SKIP() << "Requires an interactive desktop session";
  std::unique_ptr<graphics::Core> core;
  ASSERT_EQ(graphics::CreateCore(GetParam(), graphics::Core::Settings{2, true}, &core), 0);
  ASSERT_EQ(core->InitializeLogicalDeviceAutoSelect(false), 0);
  std::unique_ptr<graphics::Window> window;
  ASSERT_EQ(core->CreateWindowObject(320, 240, "HDR reference white test", &window), 0);
  if (!HasHDRSurface(window.get()))
    GTEST_SKIP() << "Desktop exposes no supported HDR surface";
  ASSERT_EQ(window->SetHDR(true), 0);
  auto *composition = window->PrepareHDRComposition(core.get(), {13, 7});
  ASSERT_NE(composition, nullptr);
  std::vector<uint16_t> source(13 * 7 * 4);
  for (size_t i = 0; i < source.size(); i += 4) {
    source[i] = glm::packHalf1x16(0.18f);
    source[i + 1] = glm::packHalf1x16(1.0f);
    source[i + 2] = glm::packHalf1x16(4.0f);
    source[i + 3] = glm::packHalf1x16(0.5f);
  }
  composition->UploadData(source.data());
  const auto brightness = window->GetDisplayBrightness();
  EXPECT_TRUE(std::isfinite(brightness.hdr_headroom));
  EXPECT_TRUE(brightness.hdr_headroom == 0.0f || brightness.hdr_headroom >= 1.0f);
  RecordProperty("hdr_headroom", std::to_string(brightness.hdr_headroom));
  const float scale = window->HDRReferenceWhiteScale();
  ASSERT_GT(scale, 0.0f);
  std::unique_ptr<graphics::CommandContext> commands;
  core->CreateCommandContext(&commands);
  auto *aligned = window->AlignHDRComposition(commands.get());
  ASSERT_EQ(core->SubmitCommandContext(commands.get()), 0);
  core->WaitGPU();
  std::vector<uint16_t> actual(source.size()), unchanged(source.size());
  aligned->DownloadData(actual.data());
  composition->DownloadData(unchanged.data());
  EXPECT_EQ(source, unchanged);
  for (size_t i = 0; i < source.size(); i += 4) {
    const auto expected = ExpectedOutput(
        window.get(),
        {glm::unpackHalf1x16(source[i]), glm::unpackHalf1x16(source[i + 1]), glm::unpackHalf1x16(source[i + 2])});
    for (size_t c = 0; c < 3; ++c)
      EXPECT_NEAR(glm::unpackHalf1x16(actual[i + c]), expected[c], 0.004f * scale);
    EXPECT_EQ(actual[i + 3], source[i + 3]);
  }
  window->SetHDRBrightnessAlignment(false);
  EXPECT_EQ(window->HDRReferenceWhiteScale(), 1.0f);
  if (UsesPQSwapchain(window.get()))
    EXPECT_NE(window->PrepareHDRComposition(core.get(), {13, 7}), nullptr);
  else
    EXPECT_EQ(window->PrepareHDRComposition(core.get(), {13, 7}), nullptr);
  window->SetHDRBrightnessAlignment(true);
  ASSERT_EQ(window->SetHDR(false), 0);
  EXPECT_EQ(window->HDRReferenceWhiteScale(), 1.0f);
  window->CloseWindow();
}

TEST_P(HDRWindowTest, UnsupportedHDRLeavesSDRPresentationUsable) {
  if (!InteractiveHDRTests())
    GTEST_SKIP();
  std::unique_ptr<graphics::Core> core;
  ASSERT_EQ(graphics::CreateCore(GetParam(), graphics::Core::Settings{2, true}, &core), 0);
  ASSERT_EQ(core->InitializeLogicalDeviceAutoSelect(false), 0);
  std::unique_ptr<graphics::Window> window;
  ASSERT_EQ(core->CreateWindowObject(320, 240, "SDR compatibility test", &window), 0);
  if (HasHDRSurface(window.get()))
    GTEST_SKIP() << "Requires an SDR-only surface";
  window->InitImGui();
  ImGui::GetIO().IniFilename = nullptr;
  for (int i = 0; i < 3; ++i) {
    EXPECT_NE(window->SetHDR(true), 0);
    window->BeginImGuiFrame();
    ImGui::TextUnformatted("SDR remains usable");
    window->EndImGuiFrame();
    std::unique_ptr<graphics::Image> image;
    ASSERT_EQ(core->CreateImage(320, 240, graphics::IMAGE_FORMAT_R8G8B8A8_UNORM, &image), 0);
    std::unique_ptr<graphics::CommandContext> commands;
    ASSERT_EQ(core->CreateCommandContext(&commands), 0);
    commands->CmdClearImage(image.get(), {{0.2f, 0.4f, 0.6f, 1.0f}});
    commands->CmdPresent(window.get(), image.get());
    ASSERT_EQ(core->SubmitCommandContext(commands.get()), 0);
    core->WaitGPU();
    window->Resize(352 + i * 16, 256);
    glfwPollEvents();
  }
  window->CloseWindow();
}

TEST_P(HDRWindowTest, PQReadbackUsesAbsoluteNitsAndPreservesSource) {
  if (!InteractiveHDRTests())
    GTEST_SKIP();
  std::unique_ptr<graphics::Core> core;
  ASSERT_EQ(graphics::CreateCore(GetParam(), graphics::Core::Settings{2, true}, &core), 0);
  ASSERT_EQ(core->InitializeLogicalDeviceAutoSelect(false), 0);
  BrightnessProbeWindow window;
  window.pq_output = true;
  ASSERT_EQ(window.SetHDR(true), 0);
  // This numeric test works even on an SDR desktop, without presenting PQ there.
  const std::vector<glm::vec4> samples{{0, 0, 0, 0.25}, {1, 1, 1, 0.5}, {10, 10, 10, 1},         {1, 0, 0, 1},
                                       {0, 1, 0, 1},    {0, 0, 1, 1},   {65504, 65504, 65504, 1}};
  std::vector<uint16_t> source;
  for (const auto &sample : samples)
    for (int c = 0; c < 4; ++c)
      source.push_back(glm::packHalf1x16(sample[c]));
  for (bool alignment : {false, true}) {
    window.SetHDRBrightnessAlignment(alignment);
    auto *composition = window.PrepareHDRComposition(core.get(), {uint32_t(samples.size()), 1});
    ASSERT_NE(composition, nullptr);
    composition->UploadData(source.data());
    std::unique_ptr<graphics::CommandContext> commands;
    ASSERT_EQ(core->CreateCommandContext(&commands), 0);
    auto *encoded = window.AlignHDRComposition(commands.get());
    ASSERT_EQ(core->SubmitCommandContext(commands.get()), 0);
    core->WaitGPU();
    std::vector<uint16_t> actual(source.size()), unchanged(source.size());
    encoded->DownloadData(actual.data());
    composition->DownloadData(unchanged.data());
    EXPECT_EQ(source, unchanged);
    for (size_t i = 0; i < samples.size(); ++i) {
      const auto expected = ExpectedPQ(glm::vec3(samples[i]));
      for (int c = 0; c < 3; ++c)
        EXPECT_NEAR(glm::unpackHalf1x16(actual[i * 4 + c]), expected[c], 0.001);
      EXPECT_EQ(actual[i * 4 + 3], source[i * 4 + 3]);
    }
    EXPECT_NEAR(glm::unpackHalf1x16(actual[4]), PQ(203.0), 0.001);
  }
  // PQ source white is not the compositor's output white. Applying the output
  // reference-white scale here as well would duplicate compositor mapping.
  for (float output_white : {80.0f, 203.0f, 400.0f}) {
    window.brightness = {output_white, output_white / 80.0f, 0.0f, true, true};
    window.RefreshDisplayBrightness();
    std::unique_ptr<graphics::CommandContext> commands;
    ASSERT_EQ(core->CreateCommandContext(&commands), 0);
    auto *encoded = window.AlignHDRComposition(commands.get());
    ASSERT_EQ(core->SubmitCommandContext(commands.get()), 0);
    core->WaitGPU();
    std::vector<uint16_t> actual(source.size());
    encoded->DownloadData(actual.data());
    EXPECT_NEAR(glm::unpackHalf1x16(actual[4]), PQ(203.0), 0.001);
  }
  // Reuse the same pipeline with scRGB: the PQ white must not leak into the
  // Windows-style scaling path, and disabling alignment still bypasses it.
  window.pq_output = false;
  window.brightness = {240.0f, 3.0f, 0.0f, true, true};
  window.RefreshDisplayBrightness();
  auto *composition = window.PrepareHDRComposition(core.get(), {uint32_t(samples.size()), 1});
  std::unique_ptr<graphics::CommandContext> commands;
  ASSERT_EQ(core->CreateCommandContext(&commands), 0);
  auto *linear = window.AlignHDRComposition(commands.get());
  ASSERT_EQ(core->SubmitCommandContext(commands.get()), 0);
  core->WaitGPU();
  std::vector<uint16_t> actual(source.size()), unchanged(source.size());
  linear->DownloadData(actual.data());
  composition->DownloadData(unchanged.data());
  EXPECT_EQ(source, unchanged);
  for (size_t i = 0; i < samples.size(); ++i) {
    for (int c = 0; c < 3; ++c)
      EXPECT_NEAR(glm::unpackHalf1x16(actual[i * 4 + c]), std::min(samples[i][c] * 3.0f, 65504.0f), 0.01f);
    EXPECT_EQ(actual[i * 4 + 3], source[i * 4 + 3]);
  }
  window.SetHDRBrightnessAlignment(false);
  EXPECT_EQ(window.PrepareHDRComposition(core.get(), {uint32_t(samples.size()), 1}), nullptr);
  window.CloseWindow();
}

#if defined(LONGMARCH_D3D12_ENABLED)
INSTANTIATE_TEST_SUITE_P(D3D12, HDRWindowTest, testing::Values(graphics::BACKEND_API_D3D12));
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
INSTANTIATE_TEST_SUITE_P(Vulkan, HDRWindowTest, testing::Values(graphics::BACKEND_API_VULKAN));
#endif
#endif
