#include <gtest/gtest.h>

#include <cstdlib>
#include "grassland/rhi/rhix/rhix.h"
#define GLFW_EXPOSE_NATIVE_COCOA
#import <Cocoa/Cocoa.h>
#include <GLFW/glfw3native.h>

TEST(RhiCocoaHDR, AllBackendsQueryTheWindowScreen) {
  if (!std::getenv("LONGMARCH_RHI_WINDOW_TESTS"))
    GTEST_SKIP() << "Requires a desktop.";
  using namespace grassland::rhi;
  std::vector<BackendAPI> backends;
#ifdef LONGMARCH_VULKAN_ENABLED
  backends.push_back(BACKEND_API_VULKAN);
#endif
#ifdef LONGMARCH_METAL_ENABLED
  backends.push_back(BACKEND_API_METAL);
#endif
  for (auto backend : backends) {
    SCOPED_TRACE(int(backend));
    auto core = grassland::rhix::CreateCore(backend);
    ASSERT_TRUE(core);
    auto device = core.CreateDeviceByAutoSelection();
    ASSERT_TRUE(device);
    auto window = device.CreateGLFWWindow({320, 240, "Shared Cocoa HDR query"});
    ASSERT_TRUE(window);
    for (const auto &option : window.GetSupportedPresentationOptions()) {
      ASSERT_TRUE(window.SetColorSpace(option.color_space));
      auto screen = [glfwGetCocoaWindow(window.GLFWWindow()) screen];
      ASSERT_NE(screen, nil);
      const float expected = screen.maximumExtendedDynamicRangeColorComponentValue;
      EXPECT_FLOAT_EQ(window.GetHDRHeadroom(), expected);
      EXPECT_FLOAT_EQ(window.GetHDRReferenceWhiteScale(), 1.0f);
      float nits = -1.0f;
      EXPECT_FALSE(window.GetHDRReferenceWhiteNits(nits));
      EXPECT_FLOAT_EQ(nits, -1.0f);
      std::cout << "HDR query: backend=" << int(backend) << " color_space=" << int(option.color_space)
                << " headroom=" << window.GetHDRHeadroom() << " NSScreen=" << expected
                << " white_scale=" << window.GetHDRReferenceWhiteScale() << " nits=unavailable\n";
    }
  }
}
