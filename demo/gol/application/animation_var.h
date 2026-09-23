#pragma once

#include <cstdint>

enum class AnimationStyle : uint32_t { kLinear = 0, kPower2, kPower5 };

class AnimationVar {
 public:
  explicit AnimationVar(float val = 0.0, AnimationStyle ani_style = AnimationStyle::kLinear);

  void UpdateTarget(float new_target);

  void TryUpdateTarget(float new_target);

  void AddTarget(float delta);

  bool Update(float t);

  [[nodiscard]] float Value() const;

  [[nodiscard]] bool IsFinished() const;

  explicit operator float() const;

 private:
  float target_{0.0};
  float origin_{0.0};
  float alpha_{1.0};
  AnimationStyle animation_style_{AnimationStyle::kLinear};
};
