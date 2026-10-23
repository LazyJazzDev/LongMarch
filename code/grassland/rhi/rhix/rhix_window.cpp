#include "grassland/rhi/rhix/rhix_window.h"

namespace grassland::rhix {
Window::Window() noexcept = default;

Window::Window(rhi::IWindow *object) noexcept : ObjectContainer(object) {
}

rhi::IWindow *Window::Get() const noexcept {
  return static_cast<rhi::IWindow *>(GetObject());
}

bool Window::PollEvents() {
  return rhi::PollEvents();
}

GLFWwindow *Window::GLFWWindow() const {
#if !defined(NDEBUG)
  if (!*this)
    return nullptr;
#endif
  return Get()->GLFWWindow();
}

bool Window::ShouldClose() const {
#if !defined(NDEBUG)
  if (!*this)
    return true;
#endif
  return Get()->ShouldClose();
}

bool Window::IsKeyDown(int key) const {
#if !defined(NDEBUG)
  if (!*this)
    return false;
#endif
  return Get()->IsKeyDown(key);
}

bool Window::SetTitle(const std::string &title) const {
#if !defined(NDEBUG)
  if (!*this)
    return false;
#endif
  return Get()->SetTitle(title);
}

Image Window::GetNextImage() const {
#if !defined(NDEBUG)
  if (!*this)
    return nullptr;
#endif
  return Get()->GetNextImage();
}

bool Window::IsSwapChainInvalid() const {
#if !defined(NDEBUG)
  if (!*this)
    return true;
#endif
  return Get()->IsSwapChainInvalid();
}

bool Window::RebuildSwapChain() const {
#if !defined(NDEBUG)
  if (!*this)
    return false;
#endif
  return Get()->RebuildSwapChain();
}

bool Window::SetPresentMode(rhi::PresentMode mode) const {
#if !defined(NDEBUG)
  if (!*this)
    return false;
#endif
  return Get()->SetPresentMode(mode);
}

rhi::PresentMode Window::GetPresentMode() const {
#if !defined(NDEBUG)
  if (!*this)
    return rhi::PRESENT_MODE_VSYNC;
#endif
  return Get()->GetPresentMode();
}

bool Window::SetColorSpace(rhi::ColorSpace color_space) const {
#if !defined(NDEBUG)
  if (!*this)
    return false;
#endif
  return Get()->SetColorSpace(color_space);
}

rhi::ColorSpace Window::GetColorSpace() const {
#if !defined(NDEBUG)
  if (!*this)
    return rhi::COLOR_SPACE_SRGB_NONLINEAR;
#endif
  return Get()->GetColorSpace();
}

std::vector<rhi::WindowPresentationOption> Window::GetSupportedPresentationOptions() const {
#if !defined(NDEBUG)
  if (!*this)
    return std::vector<rhi::WindowPresentationOption>{};
#endif
  return Get()->GetSupportedPresentationOptions();
}
}  // namespace grassland::rhix
