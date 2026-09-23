#include "animation_var.h"

#include "interpolation.h"

AnimationVar::AnimationVar(float val, AnimationStyle ani_style)
    : target_(val),
      origin_(val),
      alpha_(1.0),
      animation_style_(ani_style) {
}

bool AnimationVar::Update(float t) {
  if (alpha_ == 1.0)
    return false;
  alpha_ += t;
  if (alpha_ >= 1.0) {
    origin_ = target_;
    alpha_ = 1.0;
  }
  return true;
}

AnimationVar::operator float() const {
  return Value();
}

void AnimationVar::UpdateTarget(float new_target) {
  origin_ = operator float();
  alpha_ = 0.0;
  target_ = new_target;
}

bool AnimationVar::IsFinished() const {
  return alpha_ == 1.0;
}

void AnimationVar::AddTarget(float delta) {
  UpdateTarget(target_ + delta);
}

void AnimationVar::TryUpdateTarget(float new_target) {
  if (new_target != target_) {
    UpdateTarget(new_target);
  }
}

float AnimationVar::Value() const {
  switch (animation_style_) {
    case AnimationStyle::kPower2:
      return Mix(origin_, target_, PowerInterpolation(alpha_, 2.0f));
    case AnimationStyle::kPower5:
      return Mix(origin_, target_, PowerInterpolation(alpha_, 5.0f));
    default:
      return Mix(origin_, target_, alpha_);
  }
}
