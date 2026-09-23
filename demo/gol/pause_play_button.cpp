#include "pause_play_button.h"

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
            {-0.24 / tan30, 0.48},
            {-0.24 / tan30, 0.0},
            {0.24 / tan30, 0.0},
            {0.24 / tan30, 0.0},
            {-0.24 / tan30, -0.48},
            {-0.24 / tan30, 0.0},
            {0.24 / tan30, 0.0},
            {0.24 / tan30, 0.0},
        },
        {
            {0.36, 0.48},
            {0.12, 0.48},
            {0.12, -0.48},
            {0.36, -0.48},
            {-0.36, 0.48},
            {-0.12, 0.48},
            {-0.12, -0.48},
            {-0.36, -0.48},
        },
        {
            {0.24 / tan30, 0.0},
            {0.24 / tan30, 0.0},
            {-0.24 / tan30, 0.0},
            {-0.24 / tan30, -0.48},
            {0.24 / tan30, 0.0},
            {0.24 / tan30, 0.0},
            {-0.24 / tan30, 0.0},
            {-0.24 / tan30, 0.48},
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

    std::vector<std::vector<Vertex>> vertices = {
        ComposeVertices(positions[0], play_color), ComposeVertices(positions[1], pause_color),
        ComposeVertices(positions[2], play_color), ComposeVertices(positions[3], pause_color),
        ComposeVertices(positions[4], play_color), ComposeVertices(positions[5], pause_color)};

    pause_play_model_ = std::make_unique<MixModel>(vertices, indices);
  }

  background_color_ = button_palette::Background();

  pause_play_device_model_ =
      std::make_unique<DeviceModel>(application_, pause_play_model_->GetModel(0.0, MixStyle::kLinear));
  pause_play_animation_var_ = AnimationVar(0.0, AnimationStyle::kPower5);

  background_animation_var_ = AnimationVar(0.0, AnimationStyle::kPower5);
}

void PausePlayButton::Update(float t) {
  pause_play_animation_var_.Update(t * 5.0f);
  pause_play_device_model_->UploadVertices(
      pause_play_model_
          ->GetModel(float(pause_play_animation_var_),
                     (click_cnt_ >= 4) ? MixStyle::kLinear : MixStyle::kAngularClockwise)
          .Vertices());
  background_animation_var_.Update(t * 10.0f);
}

void PausePlayButton::OnClick() {
  click_cnt_++;
  click_cnt_ %= 6;
  pause_play_animation_var_.AddTarget(1.0);
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
  return click_cnt_ & 1;
}
