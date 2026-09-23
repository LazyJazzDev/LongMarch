#pragma once
#include <array>
#include <random>

#include "application/animation_var.h"
#include "application/button.h"
#include "application/model.h"

class RandomizeButton : public Button {
 public:
  RandomizeButton(Application *app, std::vector<uint8_t> *cells, DeviceModel *background);
  void Update(float delta_time);
  void Draw();

 private:
  void OnClick() override;
  void OnStateChange(int state) override;

  std::vector<uint8_t> *cells_;
  DeviceModel *background_;
  std::unique_ptr<DeviceModel> face_;
  std::array<std::unique_ptr<DeviceModel>, 6> pips_;
  std::mt19937 random_engine_{std::random_device{}()};
  AnimationVar rotation_{0.0f, AnimationStyle::kPower2};
  AnimationVar background_animation_{0.0f, AnimationStyle::kPower5};
  MixValue<glm::vec4> background_color_;
};
