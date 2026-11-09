#include "grassland/rhi/backend/backend_window.h"

#include <cmath>
#define GLFW_EXPOSE_NATIVE_COCOA
#import <Cocoa/Cocoa.h>
#include <GLFW/glfw3native.h>
#import <QuartzCore/QuartzCore.h>

namespace grassland::rhi::backend {

void glfw::DestroyCocoaWindow(GLFWwindow *window) {
  @autoreleasepool {
    // Closing must not leave an animated window onscreen when the caller stops pumping events.
    [glfwGetCocoaWindow(window) setAnimationBehavior:NSWindowAnimationBehaviorNone];
    glfwDestroyWindow(window);
    [NSApp updateWindows];
    [CATransaction flush];
  }
}

float Window::HDRHeadroom() const {
  if (!glfw::IsWindowThread())
    return 10.0f;
  @autoreleasepool {
    auto screen = [glfwGetCocoaWindow(GLFWWindow()) screen];
    if (screen) {
      const float headroom = screen.maximumExtendedDynamicRangeColorComponentValue;
      if (std::isfinite(headroom) && headroom >= 1.0f)
        return headroom;
    }
    return 10.0f;
  }
}

}  // namespace grassland::rhi::backend
