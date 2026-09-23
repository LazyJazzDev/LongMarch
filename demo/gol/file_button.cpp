#include "file_button.h"

#include "button_palette.h"

namespace {
// Stroke geometry uses the same model/shader path as the existing toolbar icons.
class Icon {
 public:
  void Line(glm::vec2 from, glm::vec2 to, float width = 0.095f) {
    auto offset = glm::normalize(glm::vec2{from.y - to.y, to.x - from.x}) * width * 0.5f;
    uint32_t base = uint32_t(vertices.size());
    for (auto p : {from + offset, to + offset, to - offset, from - offset})
      vertices.push_back({p, glm::vec4{1.0f}});
    for (uint32_t i : {0u, 1u, 2u, 0u, 2u, 3u})
      indices.push_back(base + i);
  }

  std::unique_ptr<DeviceModel> Build(Application *app) {
    return std::make_unique<DeviceModel>(app, Model(vertices, indices));
  }

 private:
  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;
};
}  // namespace

FileButton::FileButton(Application *app, DeviceModel *background, Kind kind, std::function<void()> on_click)
    : Button(app, 0, 0, 100, 100),
      background_(background),
      kind_(kind),
      on_click_(std::move(on_click)),
      background_color_(button_palette::Background()) {
  Icon frame;
  if (kind == Kind::kOpen) {
    // Open folder, with a raised tab and a sloping front flap.
    frame.Line({-0.50f, 0.42f}, {-0.50f, -0.38f});
    frame.Line({-0.50f, -0.38f}, {-0.24f, -0.38f});
    frame.Line({-0.24f, -0.38f}, {-0.10f, -0.23f});
    frame.Line({0.34f, -0.23f}, {0.44f, -0.23f});
    frame.Line({0.44f, -0.23f}, {0.44f, -0.05f});
    frame.Line({-0.50f, 0.42f}, {0.37f, 0.42f});
    frame.Line({0.37f, 0.42f}, {0.53f, -0.02f});
    frame.Line({-0.50f, 0.42f}, {-0.34f, -0.02f});
    frame.Line({-0.34f, -0.02f}, {-0.20f, -0.02f});
    frame.Line({0.30f, -0.02f}, {0.53f, -0.02f});
  } else {
    // Save tray, leaving the center open for the descending arrow.
    frame.Line({-0.48f, 0.12f}, {-0.48f, 0.44f});
    frame.Line({-0.48f, 0.44f}, {0.48f, 0.44f});
    frame.Line({0.48f, 0.44f}, {0.48f, 0.12f});
    frame.Line({-0.48f, 0.12f}, {-0.30f, 0.12f});
    frame.Line({0.30f, 0.12f}, {0.48f, 0.12f});
  }
  frame_ = frame.Build(app);
  Icon arrow;
  const float direction = kind == Kind::kSave ? 1.0f : -1.0f;
  const float arrow_x = kind == Kind::kOpen ? 0.10f : 0.0f;
  arrow.Line({arrow_x, -0.42f}, {arrow_x, 0.13f});
  const float tip = direction > 0 ? 0.13f : -0.42f;
  arrow.Line({arrow_x - 0.16f, tip - direction * 0.16f}, {arrow_x, tip});
  arrow.Line({arrow_x, tip}, {arrow_x + 0.16f, tip - direction * 0.16f});
  arrow_ = arrow.Build(app);
  Icon check;
  check.Line({-0.22f, -0.16f}, {-0.06f, 0.02f}, 0.10f);
  check.Line({-0.06f, 0.02f}, {0.28f, -0.34f}, 0.10f);
  check_ = check.Build(app);
}

void FileButton::Update(float delta_time) {
  hover_.Update(delta_time * 10.0f);
  motion_ = std::min(1.0f, motion_ + delta_time / 0.65f);
  feedback_ = std::max(0.0f, feedback_ - delta_time);
}

void FileButton::Draw() {
  const glm::vec2 position{left_, top_}, size{right_ - left_, bottom_ - top_};
  const float pulse = std::sin(glm::pi<float>() * motion_);
  auto background = background_color_.GetValue(float(hover_));
  if (feedback_ > 0)
    background = glm::mix(background, success_ ? glm::vec4{0.16f, 0.25f, 0.20f, 1} : glm::vec4{0.30f, 0.13f, 0.13f, 1},
                          std::min(feedback_ * 2.0f, 0.65f));
  application_->DrawModel(background_, {GetModelMatrix(position, size, 0.6f), background, glm::uvec4{1, 0, 0, 0}});
  float shake = feedback_ > 0 && !success_ ? std::sin(feedback_ * 45.0f) * std::min(feedback_, 0.3f) * 0.035f : 0;
  auto icon_position = position + glm::vec2{shake * size.x, 0};
  static const MixValue<float> brightness({1.0f, 1.08f, 0.90f});
  auto color = button_palette::kNeutral * brightness.GetValue(float(hover_));
  application_->DrawModel(frame_.get(), {GetModelMatrix(icon_position, size, 0.4f), color, glm::uvec4{1, 0, 0, 0}});
  if (feedback_ > 0 && success_) {
    const float scale = 0.80f + 0.20f * glm::smoothstep(0.0f, 0.18f, 1.2f - feedback_);
    application_->DrawModel(check_.get(),
                            {GetModelMatrix(icon_position + size * (1.0f - scale) * 0.5f, size * scale, 0.38f),
                             button_palette::kPlay, glm::uvec4{1, 0, 0, 0}});
  } else {
    float direction = kind_ == Kind::kSave ? 1.0f : -1.0f;
    icon_position.y += direction * pulse * size.y * 0.07f;
    application_->DrawModel(arrow_.get(), {GetModelMatrix(icon_position, size, 0.38f), color, glm::uvec4{1, 0, 0, 0}});
  }
}

void FileButton::Feedback(bool success) {
  success_ = success;
  feedback_ = success ? 1.2f : 0.65f;
  motion_ = 0.0f;
}

void FileButton::BeginAction() {
  motion_ = 0.0f;
  feedback_ = 0.0f;
}

void FileButton::OnClick() {
  on_click_();
}

void FileButton::OnStateChange(int state) {
  hover_.UpdateTarget(float(state));
}
