#pragma once

#include <optional>

#include "grassland/rhi/rhi.h"

// Keep presentation changes outside event dispatch: native presentation may
// dispatch input while a frame is still being submitted.
class WindowKeyInput {
 public:
  using KeyEvent = grassland::EventManager<void(int, int, int, int)>;

  explicit WindowKeyInput(KeyEvent *events) : events_(events) {
    if (events_) {
      callback_ = events_->RegisterCallback([this](int key, int, int action, int) {
        if (action != GLFW_PRESS)
          return;
        if (key == GLFW_KEY_ESCAPE)
          close_requested_ = true;
        if (key == GLFW_KEY_P) {
          using namespace grassland::rhi;
          requested_ = requested_ == PRESENT_MODE_VSYNC ? PRESENT_MODE_CONTINUOUS : PRESENT_MODE_VSYNC;
          pending_ = requested_;
        }
      });
    }
  }

  ~WindowKeyInput() {
    if (events_)
      events_->UnregisterCallback(callback_);
  }

  WindowKeyInput(const WindowKeyInput &) = delete;
  WindowKeyInput &operator=(const WindowKeyInput &) = delete;

  bool CloseRequested() const {
    return close_requested_;
  }

  std::optional<grassland::rhi::PresentMode> TakePresentMode() {
    const auto result = pending_;
    pending_.reset();
    return result;
  }

 private:
  KeyEvent *events_;
  uint32_t callback_{};
  bool close_requested_{};
  grassland::rhi::PresentMode requested_{grassland::rhi::PRESENT_MODE_VSYNC};
  std::optional<grassland::rhi::PresentMode> pending_;
};
