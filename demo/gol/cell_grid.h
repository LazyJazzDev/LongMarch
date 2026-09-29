#pragma once

#include <vector>

#include "application/animation_var.h"
#include "application/listener.h"
#include "application/model.h"

namespace life_demo {

// Animated appearance of one cell, stored contiguously so the per-frame pass
// streams through memory.
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

// Input and animation for every cell. The cell under the cursor is found from
// the grid layout instead of one listener per cell, so pointer events and
// pan/zoom cost the same at any grid size. Each cell behaves like a button:
// hover, press, and toggle on release over the pressed cell.
class CellGrid : public Listener {
 public:
  explicit CellGrid(Application *app);
  ~CellGrid();

  // Binds the cell values and resets every visual to its value.
  void Reset(uint8_t *cells, int width, int height);

  // Top-left corner of cell (0, 0), cell pitch and clip bounds, all in
  // framebuffer pixels. Clears the hover state, as re-laying out buttons did.
  void Layout(glm::vec2 origin, float unit, glm::vec4 clip);

  // Animates every cell and refreshes Appearance(); returns whether any cell
  // still animates.
  bool Update(float t, bool animate_state);

  [[nodiscard]] const std::vector<uint32_t> &Appearance() const {
    return appearance_;
  }

  [[nodiscard]] glm::vec2 Origin() const {
    return origin_;
  }

  [[nodiscard]] float Unit() const {
    return unit_;
  }

  // While suspended, pointer input neither hovers, presses nor toggles cells;
  // suspending also drops the current hover and press, so dragging the grid
  // does not toggle the cell it started on.
  void SetSuspended(bool suspended);

  void OnCursorEnter(int enter) override;
  void OnCursorPos(double xpos, double ypos) override;
  void OnMouseButton(int mouse_button, int state, int mods) override;

 private:
  // Index of the cell whose pitch contains the window position, or -1.
  [[nodiscard]] int CellAt(glm::dvec2 window_position) const;
  void SetState(int index, int state);

  uint8_t *cells_{};
  int width_{}, height_{};
  glm::vec2 origin_{};
  float unit_{};
  glm::vec4 clip_{};
  std::vector<CellVisual> visuals_;
  std::vector<uint32_t> appearance_;
  // At most one cell is hovered or pressed: the one under the cursor.
  int active_{-1};
  int active_state_{0};
  bool suspended_{false};
};

}  // namespace life_demo
