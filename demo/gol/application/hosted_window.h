#pragma once
#include <glm/glm.hpp>
#include <set>

#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_window.h"

namespace life_demo {
// Input-only window for native hosts. Presentation stays with UIKit/ArkUI.
class HostedWindow final : public grassland::rhi::backend::Object<grassland::rhi::IWindow> {
 public:
  void UpdateHostedSize(glm::ivec2 size, glm::ivec2 framebuffer) {
    size_ = size;
    framebuffer_ = framebuffer;
    resize_.InvokeCallbacks(size.x, size.y);
    framebuffer_resize_.InvokeCallbacks(framebuffer.x, framebuffer.y);
  }

  glm::ivec2 GetSize() const {
    return size_;
  }

  glm::ivec2 GetFramebufferSize() const {
    return framebuffer_;
  }

  glm::dvec2 GetCursorPosition() const {
    return cursor_;
  }

  void SendPointer(double x, double y) {
    cursor_ = {x, y};
    mouse_move_.InvokeCallbacks(x, y);
  }

  void SendMouseButton(int button, int action, int mods = 0) {
    if (action)
      buttons_.insert(button);
    else
      buttons_.erase(button);
    mouse_button_.InvokeCallbacks(button, action, mods, cursor_.x, cursor_.y);
  }

  void SendKey(int key, int action, int mods = 0) {
    if (action)
      keys_.insert(key);
    else
      keys_.erase(key);
    key_.InvokeCallbacks(key, 0, action, mods);
  }

  void SendFocus(bool value) {
    focused_ = value;
    if (!value) {
      buttons_.clear();
      keys_.clear();
    }
    focus_.InvokeCallbacks(value);
  }

  bool IsFocused() const {
    return focused_;
  }

  bool IsMouseButtonDown(int button) const {
    return buttons_.count(button) != 0;
  }

  void SendCursorEnter(bool value) {
    cursor_enter_.InvokeCallbacks(value);
  }

#define HOST_EVENT(Name, member, ...)                           \
  grassland::EventManager<void(__VA_ARGS__)> &Name() override { \
    return member;                                              \
  }
  HOST_EVENT(ResizeEvent, resize_, int, int)
  HOST_EVENT(FramebufferResizeEvent, framebuffer_resize_, int, int)
  HOST_EVENT(CursorEnterEvent, cursor_enter_, bool)
  HOST_EVENT(FocusEvent, focus_, bool)
  HOST_EVENT(MouseMoveEvent, mouse_move_, double, double)
  HOST_EVENT(MouseButtonEvent, mouse_button_, int, int, int, double, double)
  HOST_EVENT(ScrollEvent, scroll_, double, double)
  HOST_EVENT(MagnifyEvent, magnify_, const grassland::rhi::MagnifyGesture &)
  HOST_EVENT(KeyEvent, key_, int, int, int, int)
  HOST_EVENT(CharEvent, char_, uint32_t)
  HOST_EVENT(DropEvent, drop_, int, const char **)
#undef HOST_EVENT

  bool SupportsMagnifyGestures() const override {
    return true;
  }

  bool EnableImGui(bool, const char *, float) override {
    return false;
  }

  bool UpdateImGui(const std::function<void()> &) override {
    return false;
  }

  GLFWwindow *GLFWWindow() const override {
    return nullptr;
  }

  bool ShouldClose() const override {
    return false;
  }

  bool IsKeyDown(int key) const override {
    return keys_.count(key) != 0;
  }

  bool SetTitle(const std::string &) override {
    return true;
  }

  grassland::rhi::IImage *GetNextImage() override {
    return nullptr;
  }

  bool IsSwapChainInvalid() const override {
    return true;
  }

  bool RebuildSwapChain() override {
    return false;
  }

  bool SetPresentMode(grassland::rhi::PresentMode) override {
    return false;
  }

  grassland::rhi::PresentMode GetPresentMode() const override {
    return grassland::rhi::PRESENT_MODE_VSYNC;
  }

  bool SetColorSpace(grassland::rhi::ColorSpace) override {
    return false;
  }

  grassland::rhi::ColorSpace GetColorSpace() const override {
    return grassland::rhi::COLOR_SPACE_SRGB_NONLINEAR;
  }

  bool GetHDRReferenceWhiteNits(float &) const override {
    return false;
  }

  float GetHDRReferenceWhiteScale() const override {
    return 1;
  }

  float GetHDRHeadroom() const override {
    return 1;
  }

  std::vector<grassland::rhi::WindowPresentationOption> GetSupportedPresentationOptions() const override {
    return {};
  }

 private:
  bool focused_{true};
  std::set<int> buttons_, keys_;
  glm::ivec2 size_{1}, framebuffer_{1};
  glm::dvec2 cursor_{};
  grassland::EventManager<void(int, int)> resize_, framebuffer_resize_;
  grassland::EventManager<void(bool)> cursor_enter_, focus_;
  grassland::EventManager<void(double, double)> mouse_move_, scroll_;
  grassland::EventManager<void(int, int, int, double, double)> mouse_button_;
  grassland::EventManager<void(const grassland::rhi::MagnifyGesture &)> magnify_;
  grassland::EventManager<void(int, int, int, int)> key_;
  grassland::EventManager<void(uint32_t)> char_;
  grassland::EventManager<void(int, const char **)> drop_;
};
}  // namespace life_demo
