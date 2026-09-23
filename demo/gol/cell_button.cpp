#include "cell_button.h"

CellButton::CellButton(Application *app,
                       float left,
                       float top,
                       float right,
                       float bottom,
                       uint8_t *cell,
                       DeviceModel *device_model)
    : Button(app, left, top, right, bottom),
      cell_(cell),
      device_model_(device_model) {
  background_brightness_[0] = MixValue<float>({0.2f, 0.16f, 0.12f});
  background_brightness_[1] = MixValue<float>({0.48f, 0.6f, 0.8f});

  background_animation_var_ = AnimationVar(0.0, AnimationStyle::kPower5);
  light_animation_var_ = AnimationVar(*cell_ ? 1.0f : 0.0f, AnimationStyle::kPower5);
}

void CellButton::Rebind(uint8_t *cell) {
  cell_ = cell;
  light_animation_var_ = AnimationVar(*cell_ ? 1.0f : 0.0f, AnimationStyle::kPower5);
  SetState(0);
}

void CellButton::Update(float t, bool animate_state) {
  const float brightness = *cell_ ? 1.0f : 0.0f;
  if (animate_state) {
    light_animation_var_.TryUpdateTarget(brightness);
    light_animation_var_.Update(t * 3.0f);
  } else {
    // Simulation generations switch immediately, without overlapping fades.
    light_animation_var_ = AnimationVar(brightness, AnimationStyle::kPower5);
  }
  background_animation_var_.Update(t * 10.0f);
}

void CellButton::Draw() {
  application_->DrawModel(
      device_model_,
      {GetModelMatrix(glm::vec2{left_, top_}, glm::vec2{right_ - left_, bottom_ - top_}, 0.4f), glm::vec4{1.0f},
       glm::uvec4{2u, glm::floatBitsToUint(background_brightness_[0].GetValue(float(background_animation_var_))),
                  glm::floatBitsToUint(background_brightness_[1].GetValue(float(background_animation_var_))),
                  glm::floatBitsToUint(float(light_animation_var_))},
       clip_bounds_});
}

void CellButton::OnResize() {
  ResizeModel();
}

void CellButton::OnClick() {
  *cell_ = *cell_ ? 0 : 1;
}

void CellButton::OnStateChange(int state) {
  background_animation_var_.UpdateTarget(float(state));
}

void CellButton::ResizeModel() {
}
