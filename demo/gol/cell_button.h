#pragma once

#include "application/animation_var.h"
#include "application/button.h"
#include "application/model.h"

namespace life_demo {

class CellButton : public Button {
 public:
  CellButton(Application *app,
             float left,
             float top,
             float right,
             float bottom,
             uint8_t *cell,
             DeviceModel *device_model);
  void Rebind(uint8_t *cell);
  void Update(float t, bool animate_state);
  void Draw();

  bool IsAnimating() const {
    return !background_animation_var_.IsFinished() || !light_animation_var_.IsFinished();
  }

 private:
  void OnResize() override;
  void OnClick() override;
  void OnStateChange(int state) override;
  void ResizeModel();
  MixValue<float> background_brightness_[2];
  DeviceModel *device_model_{};
  AnimationVar background_animation_var_;
  AnimationVar light_animation_var_;
  int32_t click_cnt_{0};
  uint8_t *cell_{nullptr};
};

}  // namespace life_demo
