#include "cell_grid.h"

#include <algorithm>
#include <cmath>

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

CellGrid::CellGrid(Application *app) : Listener(app) {
}

CellGrid::~CellGrid() {
  application_->UnregisterListener(this);
}

void CellGrid::Reset(uint8_t *cells, int width, int height) {
  cells_ = cells;
  width_ = width;
  height_ = height;
  visuals_.assign(size_t(width) * height, {});
  appearance_.assign(visuals_.size(), 0);
  for (size_t i = 0; i < visuals_.size(); ++i)
    visuals_[i].Reset(cells_[i]);
  active_ = -1;
  active_state_ = 0;
}

void CellGrid::Layout(glm::vec2 origin, float unit, glm::vec4 clip) {
  origin_ = origin;
  unit_ = unit;
  clip_ = clip;
  SetState(active_, 0);
  active_ = -1;
}

bool CellGrid::Update(float t, bool animate_state) {
  bool animating = false;
  for (size_t i = 0; i < visuals_.size(); ++i)
    animating |= visuals_[i].Update(cells_[i] != 0, t, animate_state, appearance_[i]);
  return animating;
}

int CellGrid::CellAt(glm::dvec2 window_position) const {
  // Pointer events use window coordinates; the layout uses framebuffer pixels.
  const auto window = application_->GetWindow();
  const glm::vec2 scale = glm::vec2(glm::max(window->GetFramebufferSize(), glm::ivec2{1})) /
                          glm::vec2(glm::max(window->GetSize(), glm::ivec2{1}));
  const glm::vec2 position = glm::vec2(window_position) * scale;
  if (position.x < clip_.x || position.x > clip_.z || position.y < clip_.y || position.y > clip_.w || unit_ <= 0.0f)
    return -1;
  const glm::vec2 grid = (position - origin_) / unit_;
  const glm::vec2 cell = glm::floor(grid);
  if (cell.x < 0.0f || cell.y < 0.0f || cell.x >= float(width_) || cell.y >= float(height_))
    return -1;
  // The whole pitch, including the gaps, selects the cell: on dense grids the
  // gaps are a large share of a fingertip.
  return int(cell.y) * width_ + int(cell.x);
}

void CellGrid::SetState(int index, int state) {
  if (index < 0)
    return;
  visuals_[index].background.UpdateTarget(float(state));
  visuals_[index].dirty = true;
  if (index == active_)
    active_state_ = state;
}

void CellGrid::OnCursorEnter(int enter) {
  if (!enter && active_ >= 0) {
    SetState(active_, 0);
    active_ = -1;
  }
}

void CellGrid::OnCursorPos(double xpos, double ypos) {
  const int index = CellAt({xpos, ypos});
  if (index == active_)
    return;
  SetState(active_, 0);
  active_ = index;
  active_state_ = 0;
  SetState(index, 1);
}

void CellGrid::OnMouseButton(int mouse_button, int state, int mods) {
  if (mouse_button != GLFW_MOUSE_BUTTON_LEFT)
    return;
  const int index = CellAt(application_->GetWindow()->GetCursorPosition());
  if (index != active_) {
    SetState(active_, 0);
    active_ = index;
    active_state_ = 0;
  }
  if (index < 0)
    return;
  if (state == GLFW_PRESS) {
    SetState(index, 2);
  } else if (state == GLFW_RELEASE) {
    if (active_state_ == 2)
      cells_[index] = cells_[index] ? 0 : 1;
    SetState(index, 1);
  }
}
