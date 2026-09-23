#pragma once

#include "application/animation_var.h"
#include "application/button.h"
#include "application/model.h"

class CellButton : public Button {
 public:
  CellButton(Application *app,
             float left,
             float top,
             float right,
             float bottom,
             uint8_t *cell,
             DeviceModel *device_model);
  void Update(float t);
  void Draw();

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
