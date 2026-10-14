#include "grassland/rhi/backend/backend_device.h"
#include "grassland/rhi/backend/backend_queue.h"
#include "grassland/rhi/backend/backend_window.h"
#include "grassland/rhi/rhi_device.h"

namespace grassland::rhi {

Window::Window() noexcept = default;

using namespace backend;

bool Window::PollEvents() {
  return glfw::PollEvents();
}

bool Window::ShouldClose() const {
  return !*this || Access::As<IWindow>(*this).ShouldClose();
}

bool Window::IsKeyDown(Key key) const {
  return *this && Access::As<IWindow>(*this).IsKeyDown(key);
}

bool Window::SetTitle(const std::string &title) const {
  return *this && Access::As<IWindow>(*this).SetTitle(title);
}

Image Window::AcquireFrame() const {
  try {
    if (*this && glfw::IsWindowThread())
      return Access::Share<Image>(Access::As<IWindow>(*this).AcquireFrame());
  } catch (const std::bad_alloc &) {
    ReportError(-1, "Frame acquisition ran out of memory.");
  }
  return {};
}

bool Window::SetPresentMode(PresentMode mode) const {
  try {
    return *this && glfw::IsWindowThread() && Access::As<IWindow>(*this).SetPresentMode(mode);
  } catch (const std::bad_alloc &) {
    ReportError(-1, "Present mode change ran out of memory.");
    return false;
  }
}

PresentMode Window::GetPresentMode() const {
  return *this ? Access::As<IWindow>(*this).GetPresentMode() : PresentMode::VSync;
}

}  // namespace grassland::rhi
