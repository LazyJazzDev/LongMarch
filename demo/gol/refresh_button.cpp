#include "refresh_button.h"

#include "button_palette.h"
#include "geometry/mesh.h"

RefreshButton::RefreshButton(Application *app,
                             float left,
                             float top,
                             float right,
                             float bottom,
                             std::vector<uint8_t> *game_buffer,
                             DeviceModel *device_model)
    : Button(app, left, top, right, bottom),
      background_device_model_(device_model) {
  game_buffer_ = game_buffer;
  ResizeModel();

  {
    std::vector<glm::vec2> refresh_logo_outline;

    const int precision = 60;
    const float start_angle = glm::radians(240.0f);
    const float end_angle = glm::radians(-40.0f);
    for (int i = 0; i <= precision; i++) {
      float angle = Mix(end_angle, start_angle, float(i) / float(precision));
      float sin_angle = std::sin(angle), cos_angle = std::cos(angle);
      refresh_logo_outline.push_back(glm::vec2{cos_angle, sin_angle} * 0.4f);
    }

    for (int i = 0; i <= precision; i++) {
      float angle = Mix(start_angle, end_angle, float(i) / float(precision));
      float sin_angle = std::sin(angle), cos_angle = std::cos(angle);
      refresh_logo_outline.push_back(glm::vec2{cos_angle, sin_angle} * 0.6f);
    }

    {
      float angle = end_angle;
      float sin_angle = std::sin(angle), cos_angle = std::cos(angle);
      auto clock_dir = glm::vec2{cos_angle, sin_angle};
      auto arrow_dir = glm::vec2{clock_dir.y, -clock_dir.x};
      auto origin = clock_dir * 0.5f;
      refresh_logo_outline.push_back(origin + clock_dir * 0.2f);
      refresh_logo_outline.push_back(origin + arrow_dir * 0.2f * std::sqrt(2.0f));
      refresh_logo_outline.push_back(origin - clock_dir * 0.2f);
    }

    auto refresh_logo_triangles = geometry::Mesh(refresh_logo_outline).GetTriangles();
    std::vector<glm::vec2> refresh_logo;
    std::vector<uint32_t> indices;
    for (auto &triangles : refresh_logo_triangles) {
      refresh_logo.emplace_back(triangles.v0);
      refresh_logo.emplace_back(triangles.v1);
      refresh_logo.emplace_back(triangles.v2);
    }

    indices.resize(refresh_logo.size());

    for (int i = 0; i < refresh_logo.size(); i++) {
      indices[i] = i;
    }

    // Match the lit speed arrows, including their icon gradient.
    auto color = button_palette::kNeutral;

    std::vector<std::vector<Vertex>> vertices = {
        ComposeVertices(refresh_logo, color),
    };

    for (auto &vertex : refresh_logo) {
      vertex.x = -vertex.x;
      vertex.y = -vertex.y;
    }

    vertices.push_back(ComposeVertices(refresh_logo, color));

    refresh_model_ = std::make_unique<MixModel>(vertices, indices);
  }

  background_color_ = button_palette::ResetBackground();

  refresh_device_model_ = std::make_unique<DeviceModel>(application_, refresh_model_->GetModel(0.0, MixStyle::kLinear));
  refresh_animation_var_ = AnimationVar(0.0, AnimationStyle::kPower5);

  background_animation_var_ = AnimationVar(0.0, AnimationStyle::kPower5);
}

void RefreshButton::Update(float t) {
  refresh_animation_var_.Update(t * 2.0f);
  refresh_device_model_->UploadVertices(
      refresh_model_->GetModel(float(refresh_animation_var_), MixStyle::kAngularClockwise).Vertices());
  background_animation_var_.Update(t * 10.0f);
}

void RefreshButton::Draw() {
  application_->DrawModel(background_device_model_,
                          {GetModelMatrix(glm::vec2{left_, top_}, glm::vec2{right_ - left_, bottom_ - top_}, 0.6f),
                           background_color_.GetValue(float(background_animation_var_)), glm::uvec4{1, 0, 0, 0}});

  application_->DrawModel(refresh_device_model_.get(),
                          {GetModelMatrix(glm::vec2{left_, top_}, glm::vec2{right_ - left_, bottom_ - top_}, 0.4f),
                           glm::vec4{1.0f}, glm::uvec4{1, 0, 0, 0}});
}

void RefreshButton::OnResize() {
  ResizeModel();
}

void RefreshButton::OnClick() {
  refresh_animation_var_.AddTarget(2.0);
  std::memset(game_buffer_->data(), 0, game_buffer_->size());
}

void RefreshButton::OnStateChange(int state) {
  background_animation_var_.UpdateTarget(float(state));
}

void RefreshButton::ResizeModel() {
}
