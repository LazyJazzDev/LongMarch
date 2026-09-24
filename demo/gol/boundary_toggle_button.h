#pragma once

#include "application/animation_var.h"
#include "application/button.h"
#include "application/model.h"
#include "game_of_life_lib/game_of_life_lib.h"

class BoundaryToggleButton : public Button {
 public:
  BoundaryToggleButton(Application *app, DeviceModel *background);
  void Update(float seconds);
  void Draw();

  BoundaryMode Mode() const {
    return mode_;
  }

  void OnClick() override;
  void OnStateChange(int state) override;

 private:
  BoundaryMode mode_{BoundaryMode::kPeriodic};
  DeviceModel *background_;
  std::unique_ptr<MixModel> icon_;
  std::unique_ptr<DeviceModel> device_icon_;
  AnimationVar hover_{0.0f, AnimationStyle::kPower5};
  float morph_{1.0f};
  float velocity_{};
};
