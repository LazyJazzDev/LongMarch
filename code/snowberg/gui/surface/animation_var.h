#pragma once

#include <cstdint>

namespace snowberg::gui::surface {

enum class AnimationStyle : uint32_t { kLinear = 0, kPower2, kPower5 };

class AnimationVar {
 public:
  explicit AnimationVar(float val = 0.0, AnimationStyle ani_style = AnimationStyle::kLinear);

  void UpdateTarget(float new_target);

  void TryUpdateTarget(float new_target);

  void AddTarget(float delta);

  bool Update(float t);

  [[nodiscard]] float Value() const;

  [[nodiscard]] bool IsFinished() const {
    return alpha_ == 1.0f;
  }

  // The value once finished; Value() equals it whenever IsFinished().
  [[nodiscard]] float Target() const {
    return target_;
  }

  explicit operator float() const;

 private:
  float target_{0.0};
  float origin_{0.0};
  float alpha_{1.0};
  AnimationStyle animation_style_{AnimationStyle::kLinear};
};

}  // namespace snowberg::gui::surface
