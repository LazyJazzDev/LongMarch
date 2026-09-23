#pragma once
#include "application/animation_var.h"
#include "application/button.h"
#include "application/model.h"

class RefreshButton : public Button {
 public:
  RefreshButton(Application *app,
                float left,
                float top,
                float right,
                float bottom,
                std::vector<uint8_t> *game_buffer,
                DeviceModel *device_model);
  void Update(float t);
  void Draw();

 private:
  void OnResize() override;
  void OnClick() override;
  void OnStateChange(int state) override;
  void ResizeModel();
  bool play_{false};
  std::unique_ptr<MixModel> refresh_model_;
  std::unique_ptr<DeviceModel> refresh_device_model_;
  AnimationVar refresh_animation_var_;
  MixValue<glm::vec4> background_color_;
  DeviceModel *background_device_model_;
  AnimationVar background_animation_var_;
  int32_t program_transformation_uniform_location_;
  std::vector<uint8_t> *game_buffer_;
};
