#pragma once
#include "application/animation_var.h"
#include "application/button.h"
#include "application/model.h"

class SpeedToggleButton : public Button {
 public:
  SpeedToggleButton(Application *app, float left, float top, float right, float bottom, DeviceModel *device_model);
  void Update(float t);
  void Draw();
  [[nodiscard]] int SpeedLevel() const;

 private:
  void OnResize() override;
  void OnClick() override;
  void OnStateChange(int state) override;
  void ResizeModel();
  bool play_{false};
  std::unique_ptr<MixModel> speed_toggle_model_;
  std::unique_ptr<DeviceModel> speed_toggle_device_model_;
  AnimationVar speed_toggle_animation_var_;
  MixValue<glm::vec4> background_color_;
  DeviceModel *background_device_model_;
  AnimationVar background_animation_var_;
  int32_t program_transformation_uniform_location_;
  int32_t click_cnt_{0};
};
