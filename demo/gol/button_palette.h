#pragma once

#include "application/model.h"

namespace button_palette {
// IconTheme multiplies these colors by roughly 1.6 at the icon center.
inline const glm::vec4 kNeutral{0.47f, 0.49f, 0.52f, 1.0f};
inline const glm::vec4 kPlay{0.39f, 0.52f, 0.43f, 1.0f};
inline const glm::vec4 kPause{0.61f, 0.45f, 0.43f, 1.0f};
inline const glm::vec4 kLightning{0.60f, 0.49f, 0.25f, 1.0f};
inline const glm::vec4 kInactive{0.25f, 0.27f, 0.30f, 1.0f};

// Dice faces use a flat shader, so include the icon gradient's gain here.
inline MixValue<glm::vec4> Dice() {
  return MixValue<glm::vec4>({{0.75f, 0.78f, 0.83f, 1.0f}, {0.86f, 0.89f, 0.94f, 1.0f}, {0.66f, 0.69f, 0.74f, 1.0f}});
}

inline MixValue<glm::vec4> Background() {
  return MixValue<glm::vec4>({{0.16f, 0.18f, 0.21f, 1.0f}, {0.20f, 0.22f, 0.25f, 1.0f}, {0.13f, 0.15f, 0.18f, 1.0f}});
}

// Keep the action hues at similar perceived brightness to the neutral background.
inline MixValue<glm::vec4> ResetBackground() {
  return MixValue<glm::vec4>(
      {{0.29f, 0.125f, 0.125f, 1.0f}, {0.34f, 0.165f, 0.165f, 1.0f}, {0.245f, 0.10f, 0.10f, 1.0f}});
}

inline MixValue<glm::vec4> RandomizeBackground() {
  return MixValue<glm::vec4>({{0.115f, 0.18f, 0.29f, 1.0f}, {0.155f, 0.22f, 0.34f, 1.0f}, {0.09f, 0.15f, 0.25f, 1.0f}});
}
}  // namespace button_palette
