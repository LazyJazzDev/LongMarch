#include <gtest/gtest.h>
#include <long_march.h>

#include <cstdlib>

#define GLFW_EXPOSE_NATIVE_COCOA
#import <Cocoa/Cocoa.h>
#include <GLFW/glfw3native.h>
#import <QuartzCore/CAMetalLayer.h>

using namespace grassland;

TEST(MetalWindowTest, HDRPresentationAndImGuiSwitching) {
  if (!std::getenv("LONGMARCH_TEST_METAL_WINDOWS"))
    GTEST_SKIP() << "Set LONGMARCH_TEST_METAL_WINDOWS=1 in an interactive macOS session";
  @autoreleasepool {
    std::unique_ptr<graphics::Core> core;
    ASSERT_EQ(graphics::CreateCore(graphics::BACKEND_API_METAL, {}, &core), 0);
    ASSERT_EQ(core->InitializeLogicalDeviceAutoSelect(false), 0);
    for (bool imgui : {false, true}) {
      std::unique_ptr<graphics::Window> window;
      ASSERT_EQ(core->CreateWindowObject(320, 240, "Metal HDR test", &window), 0);
      auto layer = (CAMetalLayer *)[glfwGetCocoaWindow(window->GLFWWindow()) contentView].layer;
      ASSERT_NE(layer, nil);
      EXPECT_EQ(layer.pixelFormat, MTLPixelFormatBGRA8Unorm);
      EXPECT_FALSE(layer.wantsExtendedDynamicRangeContent);
      if (imgui) {
        window->InitImGui();
        ImGui::GetIO().IniFilename = nullptr;
      }
      std::unique_ptr<graphics::Image> image;
      core->CreateImage(320, 240, graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT, &image);
      for (bool hdr : {true, true, false, true, false, true}) {
        window->SetHDR(hdr);
        EXPECT_EQ(layer.pixelFormat, hdr ? MTLPixelFormatRGBA16Float : MTLPixelFormatBGRA8Unorm);
        EXPECT_EQ(bool(layer.wantsExtendedDynamicRangeContent), hdr);
        auto expected = CGColorSpaceCreateWithName(hdr ? kCGColorSpaceExtendedLinearSRGB : kCGColorSpaceSRGB);
        EXPECT_TRUE(CFEqual(layer.colorspace, expected));
        CGColorSpaceRelease(expected);
        if (imgui) {
          window->BeginImGuiFrame();
          ImGui::TextUnformatted("SDR UI on HDR content");
          window->EndImGuiFrame();
        }
        std::unique_ptr<graphics::CommandContext> commands;
        core->CreateCommandContext(&commands);
        commands->CmdClearImage(image.get(), {{3.0f, 2.0f, 1.0f, 1.0f}});
        commands->CmdPresent(window.get(), image.get());
        ASSERT_EQ(core->SubmitCommandContext(commands.get()), 0);
      }
      // Closing immediately after queued presentation must remain safe in HDR mode.
      window->CloseWindow();
      window->CloseWindow();
    }
    core->WaitGPU();
  }
}
