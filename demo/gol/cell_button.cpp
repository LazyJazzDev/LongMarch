#include "cell_button.h"

#include <algorithm>
#include <cmath>

namespace life_demo {

namespace {
uint32_t Unorm10(float value) {
  return uint32_t(std::lround(std::clamp(value, 0.0f, 1.0f) * 1023.0f));
}

// Rest, hover and press brightness, shared by every cell.
const MixValue<float> &DeadBrightness() {
  static const MixValue<float> value({0.2f, 0.16f, 0.12f});
  return value;
}

const MixValue<float> &LiveBrightness() {
  static const MixValue<float> value({0.48f, 0.6f, 0.8f});
  return value;
}
}  // namespace

void CellVisual::Reset(bool alive) {
  light = AnimationVar(alive ? 1.0f : 0.0f, AnimationStyle::kPower5);
  dirty = true;
}

bool CellVisual::Animate(bool alive, float t, bool animate_state, uint32_t &packed) {
  const float brightness = alive ? 1.0f : 0.0f;
  if (animate_state) {
    light.TryUpdateTarget(brightness);
    dirty |= light.Update(t * 3.0f);
  } else if (!light.IsFinished() || float(light) != brightness) {
    // Simulation generations switch immediately, without overlapping fades.
    light = AnimationVar(brightness, AnimationStyle::kPower5);
    dirty = true;
  }
  dirty |= background.Update(t * 10.0f);
  if (dirty) {
    const float state = float(background);
    packed = Unorm10(DeadBrightness().GetValue(state)) | Unorm10(LiveBrightness().GetValue(state)) << 10 |
             Unorm10(float(light)) << 20;
    dirty = false;
  }
  return !background.IsFinished() || !light.IsFinished();
}

CellButton::CellButton(Application *app,
                       float left,
                       float top,
                       float right,
                       float bottom,
                       uint8_t *cell,
                       CellVisual *visual)
    : Button(app, left, top, right, bottom),
      cell_(cell),
      visual_(visual) {
  visual_->Reset(*cell_);
}

void CellButton::Rebind(uint8_t *cell, CellVisual *visual) {
  cell_ = cell;
  visual_ = visual;
  visual_->Reset(*cell_);
  SetState(0);
}

void CellButton::OnClick() {
  *cell_ = *cell_ ? 0 : 1;
}

void CellButton::OnStateChange(int state) {
  visual_->background.UpdateTarget(float(state));
  visual_->dirty = true;
}

}  // namespace life_demo
