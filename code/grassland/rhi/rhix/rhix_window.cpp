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
  return *this ? Get()->GLFWWindow() : nullptr;
}

bool Window::ShouldClose() const {
  return !*this || Get()->ShouldClose();
}

bool Window::IsKeyDown(int key) const {
  return *this && Get()->IsKeyDown(key);
}

bool Window::SetTitle(const std::string &title) const {
  return *this && Get()->SetTitle(title);
}

Image Window::AcquireFrame() const {
  return *this ? Get()->AcquireFrame() : nullptr;
}

bool Window::RebuildSwapChain() const {
  if (!*this)
    return false;
  return Get()->RebuildSwapChain();
}

bool Window::SetPresentMode(rhi::PresentMode mode) const {
  return *this && Get()->SetPresentMode(mode);
}

rhi::PresentMode Window::GetPresentMode() const {
  return *this ? Get()->GetPresentMode() : rhi::PRESENT_MODE_VSYNC;
}

std::vector<rhi::WindowPresentationOption> Window::GetSupportedPresentationOptions() const {
  return *this ? Get()->GetSupportedPresentationOptions() : std::vector<rhi::WindowPresentationOption>{};
}
}  // namespace grassland::rhix
