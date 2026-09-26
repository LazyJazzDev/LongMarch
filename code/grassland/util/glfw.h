#pragma once

#include <GLFW/glfw3.h>

#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace grassland {

// Use the same selection for Vulkan's temporary initialization and real windows.
// Keep GLFW's native selection on other operating systems and older GLFW builds.
inline bool InitializeGLFWWithPlatform() {
#if defined(__linux__) && defined(GLFW_PLATFORM)
  const char *platform = std::getenv("LONGMARCH_WINDOW_SYSTEM");
  if (platform && std::strcmp(platform, "auto") != 0) {
    int requested;
    if (std::strcmp(platform, "x11") == 0)
      requested = GLFW_PLATFORM_X11;
    else if (std::strcmp(platform, "wayland") == 0)
      requested = GLFW_PLATFORM_WAYLAND;
    else
      throw std::invalid_argument("LONGMARCH_WINDOW_SYSTEM must be auto, x11 or wayland");
    if (!glfwPlatformSupported(requested))
      throw std::runtime_error("Requested window system was not compiled into GLFW");
    glfwInitHint(GLFW_PLATFORM, requested);
    return glfwInit() == GLFW_TRUE;
  }
  // A Wayland-enabled binary must still run on X11-only desktops. A stale
  // WAYLAND_DISPLAY must not prevent using an available X server either.
  if (std::getenv("WAYLAND_DISPLAY") && glfwPlatformSupported(GLFW_PLATFORM_WAYLAND)) {
    glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_WAYLAND);
    if (glfwInit())
      return true;
    glfwGetError(nullptr);
  }
  glfwInitHint(GLFW_PLATFORM, std::getenv("DISPLAY") && glfwPlatformSupported(GLFW_PLATFORM_X11) ? GLFW_PLATFORM_X11
                                                                                                 : GLFW_ANY_PLATFORM);
#endif
  return glfwInit() == GLFW_TRUE;
}

}  // namespace grassland
