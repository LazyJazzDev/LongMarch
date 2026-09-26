#include <gtest/gtest.h>
#include <long_march.h>

#include <cstdlib>
#include <glm/gtc/packing.hpp>

#if defined(LONGMARCH_D3D12_ENABLED)
#include "grassland/graphics/backend/d3d12/d3d12_window.h"
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
#include "grassland/graphics/backend/vulkan/vulkan_window.h"
#endif

using namespace grassland;

#if defined(LONGMARCH_VULKAN_ENABLED)
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
#endif

class BrightnessProbeWindow : public graphics::Window {
 public:
  BrightnessProbeWindow() : Window(64, 64, "Brightness query test", false, false, false) {
  }

  graphics::DisplayBrightness brightness;

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
  graphics::DisplayBrightness QueryDisplayBrightness() const override {
    return brightness;
  }
};

TEST(HDRBrightnessTest, RefreshNotifiesAndUnknownReferenceFallsBack) {
  if (!std::getenv("LONGMARCH_TEST_HDR_WINDOWS"))
    GTEST_SKIP();
  BrightnessProbeWindow window;
  int changes = 0;
  window.DisplayBrightnessEvent().RegisterCallback([&](const graphics::DisplayBrightness &) { ++changes; });
  window.brightness = {480.0f, 6.0f, 0.0f, true, true};
  window.RefreshDisplayBrightness();
  EXPECT_EQ(changes, 1);
  EXPECT_EQ(window.HDRReferenceWhiteScale(), 1.0f);  // SDR must never scale.
  window.SetHDR(true);
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
  // Capability-only changes must notify even when reference white is unchanged.
  window.brightness.max_luminance_nits = 1000.0f;
  window.brightness.reported_luminance_known = true;
  window.RefreshDisplayBrightness();
  EXPECT_EQ(changes, 4);
  window.brightness.max_full_frame_luminance_nits = 400.0f;
  window.RefreshDisplayBrightness();
  EXPECT_EQ(changes, 5);
  window.RefreshDisplayBrightness();
  EXPECT_EQ(changes, 5);
}

#if defined(LONGMARCH_D3D12_ENABLED) || defined(LONGMARCH_VULKAN_ENABLED)
class HDRWindowTest : public testing::TestWithParam<graphics::BackendAPI> {};

TEST_P(HDRWindowTest, PresentationAndImGuiSwitching) {
  if (!std::getenv("LONGMARCH_TEST_HDR_WINDOWS"))
    GTEST_SKIP() << "Set LONGMARCH_TEST_HDR_WINDOWS=1 in an interactive desktop session";
  std::unique_ptr<graphics::Core> core;
  ASSERT_EQ(graphics::CreateCore(GetParam(), graphics::Core::Settings{2, true}, &core), 0);
  ASSERT_EQ(core->InitializeLogicalDeviceAutoSelect(false), 0);
  for (bool imgui : {false, true}) {
    std::unique_ptr<graphics::Window> window;
    ASSERT_EQ(core->CreateWindowObject(320, 240, "HDR presentation test", &window), 0);
    if (imgui) {
      window->InitImGui();
      ImGui::GetIO().IniFilename = nullptr;
    }
    std::unique_ptr<graphics::Image> image;
    ASSERT_EQ(core->CreateImage(320, 240, graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT, &image), 0);
    bool resized = false;
    for (bool hdr : {true, true, false, true, false}) {
      window->SetHDR(hdr);
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
        EXPECT_EQ(native->SwapChain()->Format(), hdr ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM);
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
        EXPECT_NEAR(glm::unpackHalf1x16(pixels[center]), (imgui ? 0.216f : 3.0f) * scale, 0.01f * scale);
        EXPECT_NEAR(glm::unpackHalf1x16(pixels[center + 1]), (imgui ? 0.216f : 2.0f) * scale, 0.01f * scale);
        EXPECT_NEAR(glm::unpackHalf1x16(pixels[center + 2]), (imgui ? 0.216f : 1.0f) * scale, 0.01f * scale);
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
  if (!std::getenv("LONGMARCH_TEST_HDR_WINDOWS"))
    GTEST_SKIP() << "Requires an interactive desktop session";
  std::unique_ptr<graphics::Core> core;
  ASSERT_EQ(graphics::CreateCore(GetParam(), graphics::Core::Settings{2, true}, &core), 0);
  ASSERT_EQ(core->InitializeLogicalDeviceAutoSelect(false), 0);
  std::unique_ptr<graphics::Window> window;
  ASSERT_EQ(core->CreateWindowObject(320, 240, "HDR reference white test", &window), 0);
  window->SetHDR(true);
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
#ifdef _WIN32
  if (brightness.reported_luminance_known) {
    EXPECT_GT(brightness.max_luminance_nits, 0.0f);
    RecordProperty("reported_peak_nits", brightness.max_luminance_nits);
    RecordProperty("sdr_white_nits", brightness.sdr_white_nits);
    if (brightness.reference_white_known) {
      EXPECT_TRUE(brightness.hdr_headroom_estimated);
      EXPECT_FLOAT_EQ(brightness.hdr_headroom,
                      std::max(1.0f, brightness.max_luminance_nits / brightness.sdr_white_nits));
    }
  } else {
    EXPECT_EQ(brightness.hdr_headroom, 0.0f);
    EXPECT_FALSE(brightness.hdr_headroom_estimated);
  }
#endif
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
    for (size_t c = 0; c < 3; ++c)
      EXPECT_NEAR(glm::unpackHalf1x16(actual[i + c]), glm::unpackHalf1x16(source[i + c]) * scale, 0.004f * scale);
    EXPECT_EQ(actual[i + 3], source[i + 3]);
  }
  window->SetHDRBrightnessAlignment(false);
  EXPECT_EQ(window->HDRReferenceWhiteScale(), 1.0f);
  EXPECT_EQ(window->PrepareHDRComposition(core.get(), {13, 7}), nullptr);
  window->SetHDRBrightnessAlignment(true);
  window->SetHDR(false);
  EXPECT_EQ(window->HDRReferenceWhiteScale(), 1.0f);
  window->CloseWindow();
}

#if defined(LONGMARCH_D3D12_ENABLED)
INSTANTIATE_TEST_SUITE_P(D3D12, HDRWindowTest, testing::Values(graphics::BACKEND_API_D3D12));
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
INSTANTIATE_TEST_SUITE_P(Vulkan, HDRWindowTest, testing::Values(graphics::BACKEND_API_VULKAN));
#endif
#endif
