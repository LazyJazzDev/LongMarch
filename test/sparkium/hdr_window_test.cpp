#include <gtest/gtest.h>
#include <long_march.h>

#include <cstdlib>

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
        window->EndImGuiFrame();
      }
      std::unique_ptr<graphics::CommandContext> commands;
      ASSERT_EQ(core->CreateCommandContext(&commands), 0);
      commands->CmdClearImage(image.get(), {{3.0f, 2.0f, 1.0f, 1.0f}});
      commands->CmdPresent(window.get(), image.get());
      ASSERT_EQ(core->SubmitCommandContext(commands.get()), 0);
      glfwPollEvents();
    }
    window->CloseWindow();
  }
  core->WaitGPU();
}

#if defined(LONGMARCH_D3D12_ENABLED)
INSTANTIATE_TEST_SUITE_P(D3D12, HDRWindowTest, testing::Values(graphics::BACKEND_API_D3D12));
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
INSTANTIATE_TEST_SUITE_P(Vulkan, HDRWindowTest, testing::Values(graphics::BACKEND_API_VULKAN));
#endif
#endif
