#pragma once

#include "application/animation_var.h"
#include "application/button.h"
#include "application/model.h"

class PausePlayButton : public Button {
 public:
  PausePlayButton(Application *app, float left, float top, float right, float bottom, DeviceModel *device_model);

  void Update(float t);

  void Draw();

  [[nodiscard]] bool IsPlaying() const;

 private:
  void OnResize() override;

  void OnClick() override;

  void OnStateChange(int state) override;

  void ResizeModel();

  std::unique_ptr<MixModel> pause_play_model_;
  std::unique_ptr<DeviceModel> pause_play_device_model_;
  AnimationVar pause_play_animation_var_;
  MixValue<glm::vec4> background_color_;
  DeviceModel *background_device_model_;
  AnimationVar background_animation_var_;
  int32_t click_cnt_{0};
};
