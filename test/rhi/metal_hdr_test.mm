#include <gtest/gtest.h>

#include <cstdlib>
#include "grassland/rhi/rhi.h"
#define GLFW_EXPOSE_NATIVE_COCOA
#import <Cocoa/Cocoa.h>
#include <GLFW/glfw3native.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

TEST(RhiMetalHDR, LayerEncodingAndEDRFollowSwapchainTransitions) {
  if (!std::getenv("LONGMARCH_RHI_WINDOW_TESTS"))
    GTEST_SKIP() << "Requires a desktop.";
  using namespace grassland::rhi;
  ObjPtr<ICore> core = grassland::rhi::CreateCore(BACKEND_API_METAL);
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  ObjPtr<IWindow> window = device->CreateGLFWWindow({160, 120, "Metal HDR color space test"});
  ASSERT_TRUE(window);
  auto layer = (CAMetalLayer *)[glfwGetCocoaWindow(window->GLFWWindow()) contentView].layer;
  ASSERT_TRUE([layer isKindOfClass:[CAMetalLayer class]]);
  for (auto space : {COLOR_SPACE_EXTENDED_SRGB_LINEAR, COLOR_SPACE_HDR10_ST2084, COLOR_SPACE_SRGB_NONLINEAR}) {
    ASSERT_TRUE(window->SetColorSpace(space));
    EXPECT_EQ(window->GetColorSpace(), space);
    EXPECT_EQ(bool(layer.wantsExtendedDynamicRangeContent), space != COLOR_SPACE_SRGB_NONLINEAR);
    const auto format = space == COLOR_SPACE_EXTENDED_SRGB_LINEAR ? MTLPixelFormatRGBA16Float
                        : space == COLOR_SPACE_HDR10_ST2084       ? MTLPixelFormatRGB10A2Unorm
                                                                  : MTLPixelFormatBGRA8Unorm;
    EXPECT_EQ(layer.pixelFormat, format);
    EXPECT_FLOAT_EQ(window->GetHDRReferenceWhiteScale(), 1.0f);
    auto screen = [glfwGetCocoaWindow(window->GLFWWindow()) screen];
    if (screen)
      EXPECT_FLOAT_EQ(window->GetHDRHeadroom(), float(screen.maximumExtendedDynamicRangeColorComponentValue));
    else
      EXPECT_FLOAT_EQ(window->GetHDRHeadroom(), 10.0f);
    auto expected =
        CGColorSpaceCreateWithName(space == COLOR_SPACE_EXTENDED_SRGB_LINEAR ? kCGColorSpaceExtendedLinearSRGB
                                   : space == COLOR_SPACE_HDR10_ST2084       ? kCGColorSpaceITUR_2100_PQ
                                                                             : kCGColorSpaceSRGB);
    ASSERT_NE(expected, nullptr);
    EXPECT_TRUE(CFEqual(layer.colorspace, expected));
    EXPECT_FALSE(window->SetColorSpace(static_cast<ColorSpace>(-1)));
    EXPECT_EQ(window->GetColorSpace(), space);
    EXPECT_TRUE(CFEqual(layer.colorspace, expected));
    CGColorSpaceRelease(expected);
  }
}

#include "grassland/rhi/metal_bridge.h"

TEST(RhiMetalHDR, DrawableBridgeWaitsForAssistQueueRendering) {
  if (!std::getenv("LONGMARCH_RHI_WINDOW_TESTS"))
    GTEST_SKIP() << "Requires a desktop.";
  using namespace grassland::rhi;
  @autoreleasepool {
    ObjPtr<ICore> core = CreateCore(BACKEND_API_METAL);
    ASSERT_TRUE(core);
    ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
    ASSERT_TRUE(device);
    ObjPtr<IWindow> window = device->CreateGLFWWindow({160, 120, "Metal event bridge test"});
    ASSERT_TRUE(window);
    auto layer = (CAMetalLayer *)[glfwGetCocoaWindow(window->GLFWWindow()) contentView].layer;
    layer.framebufferOnly = NO;
    id<CAMetalDrawable> drawable = [layer nextDrawable];
    ASSERT_NE(drawable, nil);
    ObjPtr<IImage> source = device->CreateImage(
        {uint32_t(drawable.texture.width), uint32_t(drawable.texture.height), IMAGE_FORMAT_B8G8R8A8_UNORM});
    ASSERT_TRUE(source);
    ObjPtr<ICommandContext> commands = device->GetQueue(QUEUE_TYPE_ASSIST)->CreateCommandContext();
    ASSERT_TRUE(commands && commands->CmdClearImage(source.Get(), {1, 0, 0, 1}));
    ObjPtr<ISubmissionToken> rendered = commands->Submit();
    ASSERT_TRUE(rendered);
    EXPECT_FALSE(PresentMetalDrawable(device.Get(), source.Get(), nullptr, (__bridge void *)drawable));
    EXPECT_TRUE(PresentMetalDrawable(device.Get(), source.Get(), rendered.Get(), (__bridge void *)drawable));
    EXPECT_EQ(rendered->GetStatus(), SUBMISSION_STATUS_COMPLETED);
  }
}
