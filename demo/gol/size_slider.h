#pragma once

#include <functional>

#include "application/listener.h"
#include "application/model.h"

namespace life_demo {

class SizeSlider : public Listener {
 public:
  SizeSlider(Application *app, char label, int value, DeviceModel *rectangle, std::function<void(int)> on_change);
  ~SizeSlider();
  void Resize(glm::vec4 bounds, bool vertical);
  void Draw();
  void SetValue(int value);

  void SetActivationHandler(std::function<void()> handler) {
    on_activate_ = std::move(handler);
  }

  bool IsDragging() const {
    return dragging_;
  }

  int Value() const {
    return value_;
  }

  // Left, top, right and bottom in framebuffer pixels.
  glm::vec4 Bounds() const {
    return bounds_;
  }

  void OnMouseButton(int button, int action, int mods) override;
  void OnCursorPos(double x, double y) override;
  void OnCursorEnter(int entered) override;
  void OnFocus(bool focused) override;

 private:
  glm::vec2 FramePosition(double x, double y) const;
  bool Contains(glm::vec2 p) const;
  void DragTo(glm::vec2 p);
  void RebuildLabel();
  void RoundedRect(glm::vec2 position,
                   glm::vec2 size,
                   float radius,
                   float depth,
                   glm::vec4 color,
                   glm::vec4 clip,
                   bool recessed);

  char label_;
  int value_;
  DeviceModel *rectangle_;
  std::unique_ptr<DeviceModel> label_model_;
  std::function<void(int)> on_change_;
  glm::vec4 bounds_{0.0f};
  bool vertical_{false};
  float label_width_{};
  std::function<void()> on_activate_;
  bool activation_pressed_{};
  glm::vec2 activation_origin_{};
  bool dragging_{false};
  bool hovered_{false};
  bool focused_{false};
  uint32_t key_callback_{};
};

}  // namespace life_demo
