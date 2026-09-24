#pragma once

#include <functional>

#include "application/animation_var.h"
#include "application/button.h"
#include "application/model.h"

class FileButton : public Button {
 public:
  enum class Kind { kOpen, kSave };
  FileButton(Application *app, DeviceModel *background, Kind kind, std::function<void()> on_click);
  void Update(float delta_time);
  void Draw();
  void BeginAction();
  void Feedback(bool success);

 private:
  void OnClick() override;
  void OnStateChange(int state) override;

  DeviceModel *background_;
  Kind kind_;
  std::function<void()> on_click_;
  std::unique_ptr<DeviceModel> frame_;
  std::unique_ptr<DeviceModel> arrow_;
  std::unique_ptr<DeviceModel> check_;
  AnimationVar hover_{0.0f, AnimationStyle::kPower5};
  MixValue<glm::vec4> background_color_;
  AnimationVar check_transition_{0.0f, AnimationStyle::kPower2};
  AnimationVar success_tint_{0.0f, AnimationStyle::kPower2};
  AnimationVar error_tint_{0.0f, AnimationStyle::kPower2};
  float shake_phase_{0.0f};
  float motion_{1.0f};
  float feedback_{0.0f};
  bool success_{false};
};
