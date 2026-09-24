#include "pause_play_button.h"

#include <algorithm>
#include <cmath>

#include "application/model.h"
#include "button_palette.h"

PausePlayButton::PausePlayButton(Application *app,
                                 float left,
                                 float top,
                                 float right,
                                 float bottom,
                                 DeviceModel *device_model)
    : Button(app, left, top, right, bottom),
      background_device_model_(device_model) {
  ResizeModel();
  {
    auto tan30 = std::tan(glm::pi<float>() / 6.0f);
    std::vector<std::vector<glm::vec2>> positions = {
        {
            {-0.24 / tan30, -0.48},
            {0.0, -0.24},
            {0.0, 0.24},
            {-0.24 / tan30, 0.48},
            {0.24 / tan30, 0.0},
            {0.0, -0.24},
            {0.0, 0.24},
            {0.24 / tan30, 0.0},
        },
        {
            {-0.36, -0.48},
            {-0.12, -0.48},
            {-0.12, 0.48},
            {-0.36, 0.48},
            {0.36, -0.48},
            {0.12, -0.48},
            {0.12, 0.48},
            {0.36, 0.48},
        },
    };

    std::vector<uint32_t> indices = {0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7};

    const auto pause_color = button_palette::kPause;
    const auto play_color = button_palette::kPlay;

    std::vector<std::vector<Vertex>> vertices = {ComposeVertices(positions[0], play_color),
                                                 ComposeVertices(positions[1], pause_color)};

    pause_play_model_ = std::make_unique<MixModel>(vertices, indices);
  }

  background_color_ = button_palette::Background();

  pause_play_device_model_ =
      std::make_unique<DeviceModel>(application_, pause_play_model_->GetModel(0.0, MixStyle::kLinear));

  background_animation_var_ = AnimationVar(0.0, AnimationStyle::kPower5);
}

void PausePlayButton::Update(float t) {
  const float target = is_playing_ ? 1.0f : 0.0f;
  if (morph_ != target || morph_velocity_ != 0.0f) {
    // Exact critically damped spring: preserve velocity on reversal and keep the
    // response independent of frame rate. Most of the transition takes about 0.2 s.
    constexpr float frequency = 24.0f;
    const float dt = std::max(t, 0.0f);
    const float offset = morph_ - target;
    const float impulse = morph_velocity_ + frequency * offset;
    const float decay = std::exp(-frequency * dt);
    morph_ = target + (offset + impulse * dt) * decay;
    morph_velocity_ = (morph_velocity_ - frequency * impulse * dt) * decay;
    if (std::abs(morph_ - target) < 0.001f && std::abs(morph_velocity_) < 0.01f) {
      morph_ = target;
      morph_velocity_ = 0.0f;
    }
    pause_play_device_model_->UploadVertices(
        pause_play_model_->GetModel(std::clamp(morph_, 0.0f, 1.0f), MixStyle::kLinear).Vertices());
  }
  background_animation_var_.Update(t * 10.0f);
}

void PausePlayButton::OnClick() {
  is_playing_ = !is_playing_;
}

void PausePlayButton::Draw() {
  application_->DrawModel(background_device_model_,
                          {GetModelMatrix(glm::vec2{left_, top_}, glm::vec2{right_ - left_, bottom_ - top_}, 0.6f),
                           background_color_.GetValue(float(background_animation_var_)), glm::uvec4{1, 0, 0, 0}});

  application_->DrawModel(pause_play_device_model_.get(),
                          {GetModelMatrix(glm::vec2{left_, top_}, glm::vec2{right_ - left_, bottom_ - top_}, 0.4f),
                           glm::vec4{1.0f}, glm::uvec4{1, 0, 0, 0}});
}

void PausePlayButton::OnResize() {
  ResizeModel();
}

void PausePlayButton::OnStateChange(int state) {
  background_animation_var_.UpdateTarget(float(state));
}

void PausePlayButton::ResizeModel() {
}

bool PausePlayButton::IsPlaying() const {
  return is_playing_;
}

void PausePlayButton::SetPlaying(bool playing) {
  is_playing_ = playing;
}
