#pragma once
#include <random>

#include "application/animation_var.h"
#include "application/button.h"
#include "application/model.h"
#include "glm/gtc/quaternion.hpp"

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
  std::mt19937 random_engine_{std::random_device{}()};
  glm::quat orientation_{1, 0, 0, 0};
  glm::quat start_orientation_{1, 0, 0, 0};
  glm::quat target_orientation_{1, 0, 0, 0};
  glm::vec3 spin_axis_{1, 0, 0};
  float rotation_progress_{1.0f};
  int selected_face_{5};
  AnimationVar background_animation_{0.0f, AnimationStyle::kPower5};
  MixValue<glm::vec4> background_color_;
  MixValue<glm::vec4> face_color_;
};
