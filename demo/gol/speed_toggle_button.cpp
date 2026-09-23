#include "speed_toggle_button.h"

SpeedToggleButton::SpeedToggleButton(Application *app,
                                     float left,
                                     float top,
                                     float right,
                                     float bottom,
                                     DeviceModel *device_model)
    : Button(app, left, top, right, bottom),
      background_device_model_(device_model) {
  ResizeModel();

  {
    const auto active_color = glm::vec4(0.5, 0.5, 0.5, 1.0);
    const auto deactive_color = glm::vec4(0.1, 0.1, 0.1, 1.0);

    const float content_width = (std::sqrt(5.0f) - 1.0f) * 0.5f;
    const float content_height = 0.48;

    std::vector<std::vector<Vertex>> vertices = {
        {
            {{-content_width, content_height}, active_color},
            {{-content_width, -content_height}, active_color},
            {{-content_width / 3.0f, 0.0f}, active_color},
            {{-content_width / 3.0f, content_height}, deactive_color},
            {{-content_width / 3.0f, -content_height}, deactive_color},
            {{content_width / 3.0f, 0.0f}, deactive_color},
            {{content_width / 3.0f, content_height}, deactive_color},
            {{content_width / 3.0f, -content_height}, deactive_color},
            {{content_width, 0.0f}, deactive_color},
        },
        {
            {{-content_width, content_height}, active_color},
            {{-content_width, -content_height}, active_color},
            {{-content_width / 3.0f, 0.0f}, active_color},
            {{-content_width / 3.0f, content_height}, active_color},
            {{-content_width / 3.0f, -content_height}, active_color},
            {{content_width / 3.0f, 0.0f}, active_color},
            {{content_width / 3.0f, content_height}, deactive_color},
            {{content_width / 3.0f, -content_height}, deactive_color},
            {{content_width, 0.0f}, deactive_color},
        },
        {
            {{-content_width, content_height}, active_color},
            {{-content_width, -content_height}, active_color},
            {{-content_width / 3.0f, 0.0f}, active_color},
            {{-content_width / 3.0f, content_height}, active_color},
            {{-content_width / 3.0f, -content_height}, active_color},
            {{content_width / 3.0f, 0.0f}, active_color},
            {{content_width / 3.0f, content_height}, active_color},
            {{content_width / 3.0f, -content_height}, active_color},
            {{content_width, 0.0f}, active_color},
        },
    };

    for (auto &model : vertices) {
      for (auto &vertex : model) {
        vertex.position.x += content_width / 8.0f;
      }
    }

    std::vector<uint32_t> indices = {0, 1, 2, 3, 4, 5, 6, 7, 8};

    speed_toggle_model_ = std::make_unique<MixModel>(vertices, indices);
  }

  background_color_ = MixValue<glm::vec4>(
      {glm::vec4{glm::vec3{0.2}, 1.0}, glm::vec4{glm::vec3{0.16}, 1.0}, glm::vec4{glm::vec3{0.12}, 1.0}});

  speed_toggle_device_model_ =
      std::make_unique<DeviceModel>(application_, speed_toggle_model_->GetModel(0.0, MixStyle::kLinear));
  speed_toggle_animation_var_ = AnimationVar(0.0, AnimationStyle::kPower5);

  background_animation_var_ = AnimationVar(0.0, AnimationStyle::kPower5);
}

void SpeedToggleButton::Update(float t) {
  speed_toggle_animation_var_.Update(t * 5.0f);
  speed_toggle_device_model_->UploadVertices(
      speed_toggle_model_->GetModel(float(speed_toggle_animation_var_), MixStyle::kLinear).Vertices());
  background_animation_var_.Update(t * 10.0f);
}

void SpeedToggleButton::Draw() {
  application_->DrawModel(background_device_model_,
                          {GetModelMatrix(glm::vec2{left_, top_}, glm::vec2{right_ - left_, bottom_ - top_}, 0.6f),
                           background_color_.GetValue(float(background_animation_var_)), glm::uvec4{1, 0, 0, 0}});

  application_->DrawModel(speed_toggle_device_model_.get(),
                          {GetModelMatrix(glm::vec2{left_, top_}, glm::vec2{right_ - left_, bottom_ - top_}, 0.4f),
                           glm::vec4{1.0f}, glm::uvec4{1, 0, 0, 0}});
}

void SpeedToggleButton::OnResize() {
  ResizeModel();
}

void SpeedToggleButton::OnClick() {
  click_cnt_++;
  click_cnt_ %= 3;
  speed_toggle_animation_var_.AddTarget(1.0);
}

void SpeedToggleButton::OnStateChange(int state) {
  background_animation_var_.UpdateTarget(float(state));
}

void SpeedToggleButton::ResizeModel() {
}

int SpeedToggleButton::SpeedLevel() const {
  return click_cnt_;
}
