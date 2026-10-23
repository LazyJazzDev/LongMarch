#include "grassland/rhi/rhix/rhix_window.h"

namespace grassland::rhix {
Window::Window() noexcept = default;

Window::Window(rhi::IWindow *object) noexcept : ObjectContainer(object) {
}

rhi::IWindow *Window::Get() const noexcept {
  return static_cast<rhi::IWindow *>(GetObject());
}

bool Window::EnableImGui(bool enabled, const char *font_file_path, float font_size) const {
  return Get() && Get()->EnableImGui(enabled, font_file_path, font_size);
}

ImGuiContext *Window::GetImGuiContext() const {
  return Get() ? Get()->GetImGuiContext() : nullptr;
}

EventManager<void(int, int)> &Window::ResizeEvent() const {
  return Get()->ResizeEvent();
}

EventManager<void(int, int)> &Window::FramebufferResizeEvent() const {
  return Get()->FramebufferResizeEvent();
}

EventManager<void(bool)> &Window::CursorEnterEvent() const {
  return Get()->CursorEnterEvent();
}

EventManager<void(bool)> &Window::FocusEvent() const {
  return Get()->FocusEvent();
}

EventManager<void(double, double)> &Window::MouseMoveEvent() const {
  return Get()->MouseMoveEvent();
}

EventManager<void(int, int, int, double, double)> &Window::MouseButtonEvent() const {
  return Get()->MouseButtonEvent();
}

EventManager<void(double, double)> &Window::ScrollEvent() const {
  return Get()->ScrollEvent();
}

EventManager<void(int, int, int, int)> &Window::KeyEvent() const {
  return Get()->KeyEvent();
}

EventManager<void(uint32_t)> &Window::CharEvent() const {
  return Get()->CharEvent();
}

EventManager<void(int, const char **)> &Window::DropEvent() const {
  return Get()->DropEvent();
}

EventManager<void()> &Window::ImGuiEvent() const {
  return Get()->ImGuiEvent();
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

std::vector<rhi::WindowPresentationOption> Window::GetSupportedPresentationOptions() const {
#if !defined(NDEBUG)
  if (!*this)
    return std::vector<rhi::WindowPresentationOption>{};
#endif
  return Get()->GetSupportedPresentationOptions();
}
}  // namespace grassland::rhix
