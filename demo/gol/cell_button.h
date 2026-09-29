#pragma once

#include "application/animation_var.h"
#include "application/button.h"
#include "application/model.h"

namespace life_demo {

// Animated appearance of one cell. The grid stores these contiguously so the
// per-frame pass streams through memory instead of visiting every button.
struct CellVisual {
  AnimationVar background{0.0f, AnimationStyle::kPower5};
  AnimationVar light{0.0f, AnimationStyle::kPower5};
  bool dirty{true};

  void Reset(bool alive);

  // Advances toward the cell state and refreshes `packed` when it changes:
  // dead and live brightness for the hover/press state, and the live-cell
  // light, as three 10-bit unorm fields read by the cell grid shader.
  // Returns whether an animation is still running.
  bool Update(bool alive, float t, bool animate_state, uint32_t &packed) {
    // Most cells are settled and already packed; skip them without any calls.
    if (!dirty && background.IsFinished() && light.IsFinished() && light.Target() == (alive ? 1.0f : 0.0f))
      return false;
    return Animate(alive, t, animate_state, packed);
  }

 private:
  bool Animate(bool alive, float t, bool animate_state, uint32_t &packed);
};

class CellButton : public Button {
 public:
  CellButton(Application *app, float left, float top, float right, float bottom, uint8_t *cell, CellVisual *visual);
  void Rebind(uint8_t *cell, CellVisual *visual);

 private:
  void OnClick() override;
  void OnStateChange(int state) override;
  uint8_t *cell_{nullptr};
  CellVisual *visual_{nullptr};
};

}  // namespace life_demo
