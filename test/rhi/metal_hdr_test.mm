#include <gtest/gtest.h>

#include <cstdlib>
#include "grassland/rhi/rhix/rhix.h"
#define GLFW_EXPOSE_NATIVE_COCOA
#import <Cocoa/Cocoa.h>
#include <GLFW/glfw3native.h>
#import <QuartzCore/CAMetalLayer.h>

TEST(RhiMetalHDR, LayerEncodingAndEDRFollowSwapchainTransitions) {
  if (!std::getenv("LONGMARCH_RHI_WINDOW_TESTS"))
    GTEST_SKIP() << "Requires a desktop.";
  using namespace grassland::rhi;
  using namespace grassland::rhix;
  auto core = grassland::rhix::CreateCore(BACKEND_API_METAL);
  ASSERT_TRUE(core);
  auto device = core.CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  auto window = device.CreateGLFWWindow({160, 120, "Metal HDR color space test"});
  ASSERT_TRUE(window);
  auto layer = (CAMetalLayer *)[glfwGetCocoaWindow(window.GLFWWindow()) contentView].layer;
  ASSERT_TRUE([layer isKindOfClass:[CAMetalLayer class]]);
  for (auto space : {COLOR_SPACE_EXTENDED_SRGB_LINEAR, COLOR_SPACE_HDR10_ST2084, COLOR_SPACE_SRGB_NONLINEAR}) {
    ASSERT_TRUE(window.SetColorSpace(space));
    EXPECT_EQ(window.GetColorSpace(), space);
    EXPECT_EQ(bool(layer.wantsExtendedDynamicRangeContent), space != COLOR_SPACE_SRGB_NONLINEAR);
    const auto format = space == COLOR_SPACE_EXTENDED_SRGB_LINEAR ? MTLPixelFormatRGBA16Float
                        : space == COLOR_SPACE_HDR10_ST2084       ? MTLPixelFormatRGB10A2Unorm
                                                                  : MTLPixelFormatBGRA8Unorm;
    EXPECT_EQ(layer.pixelFormat, format);
    auto expected =
        CGColorSpaceCreateWithName(space == COLOR_SPACE_EXTENDED_SRGB_LINEAR ? kCGColorSpaceExtendedLinearSRGB
                                   : space == COLOR_SPACE_HDR10_ST2084       ? kCGColorSpaceITUR_2100_PQ
                                                                             : kCGColorSpaceSRGB);
    ASSERT_NE(expected, nullptr);
    EXPECT_TRUE(CFEqual(layer.colorspace, expected));
    EXPECT_FALSE(window.SetColorSpace(static_cast<ColorSpace>(-1)));
    EXPECT_EQ(window.GetColorSpace(), space);
    EXPECT_TRUE(CFEqual(layer.colorspace, expected));
    CGColorSpaceRelease(expected);
  }
}
