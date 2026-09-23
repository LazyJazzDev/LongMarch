#pragma once

#include <functional>

#include "application/listener.h"
#include "application/model.h"

class SizeSlider : public Listener {
 public:
  SizeSlider(Application *app, char label, int value, DeviceModel *rectangle, std::function<void(int)> on_change);
  ~SizeSlider();
  void Resize(glm::vec4 bounds, bool vertical);
  void Draw();

  bool IsDragging() const {
    return dragging_;
  }

  int Value() const {
    return value_;
  }

  void OnMouseButton(int button, int action, int mods) override;
  void OnCursorPos(double x, double y) override;
  void OnCursorEnter(int entered) override;

 private:
  glm::vec2 FramePosition(double x, double y) const;
  bool Contains(glm::vec2 p) const;
  void DragTo(glm::vec2 p);
  void SetValue(int value);
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
  bool dragging_{false};
  bool hovered_{false};
  bool focused_{false};
  uint32_t key_callback_{};
};
