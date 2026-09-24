#include "file_button.h"

#include "button_palette.h"
#include "geometry/mesh.h"

namespace {
std::unique_ptr<DeviceModel> BuildIcon(Application *app, const std::vector<glm::vec2> &outline) {
  // Triangulate one continuous silhouette, just like the reset icon. Shared
  // boundaries eliminate the gaps and overlapping ends of separate line quads.
  const geometry::Mesh mesh(outline);
  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;
  for (const auto &triangle : mesh.GetTriangles()) {
    for (auto point : {triangle.v0, triangle.v1, triangle.v2}) {
      indices.push_back(uint32_t(vertices.size()));
      vertices.push_back({point, glm::vec4{1.0f}});
    }
  }
  return std::make_unique<DeviceModel>(app, Model(vertices, indices));
}
}  // namespace

FileButton::FileButton(Application *app, DeviceModel *background, Kind kind, std::function<void()> on_click)
    : Button(app, 0, 0, 100, 100),
      background_(background),
      kind_(kind),
      on_click_(std::move(on_click)),
      background_color_(button_palette::Background()) {
  // A 0.20-wide stroke matches the reset icon's 0.40/0.60 inner/outer radii.
  if (kind == Kind::kOpen) {
    frame_ = BuildIcon(app, {{-0.10f, -0.24f},
                             {-0.28f, -0.44f},
                             {-0.58f, -0.44f},
                             {-0.58f, 0.48f},
                             {0.39f, 0.48f},
                             {0.59f, -0.08f},
                             {0.38f, -0.08f},
                             {0.25f, 0.28f},
                             {-0.38f, 0.28f},
                             {-0.38f, -0.24f},
                             {-0.36f, -0.24f},
                             {-0.24f, -0.10f}});
  } else {
    frame_ = BuildIcon(app, {{-0.58f, 0.04f},
                             {-0.38f, 0.04f},
                             {-0.38f, 0.28f},
                             {0.38f, 0.28f},
                             {0.38f, 0.04f},
                             {0.58f, 0.04f},
                             {0.58f, 0.48f},
                             {-0.58f, 0.48f}});
  }
  // Solid arrowhead and shaft form a single outline, with room for their motion.
  std::vector<glm::vec2> arrow{{-0.10f, -0.50f}, {0.10f, -0.50f},  {0.10f, -0.18f}, {0.27f, -0.18f},
                               {0.0f, 0.10f},    {-0.27f, -0.18f}, {-0.10f, -0.18f}};
  if (kind == Kind::kOpen) {
    for (auto &point : arrow) {
      point.x += 0.17f;
      point.y = -0.48f - point.y;
    }
  }
  arrow_ = BuildIcon(app, arrow);
  std::vector<glm::vec2> check{{-0.32f, -0.10f}, {-0.18f, -0.24f}, {-0.06f, -0.10f},
                               {0.23f, -0.40f},  {0.37f, -0.26f},  {-0.06f, 0.18f}};
  if (kind == Kind::kOpen)
    for (auto &point : check)
      point += glm::vec2{0.10f, 0.04f};
  check_ = BuildIcon(app, check);
}

void FileButton::Update(float delta_time) {
  hover_.Update(delta_time * 10.0f);
  motion_ = std::min(1.0f, motion_ + delta_time / 0.65f);
  feedback_ = std::max(0.0f, feedback_ - delta_time);
  const bool active = feedback_ > 0.0f;
  check_transition_.TryUpdateTarget(active && success_ ? 1.0f : 0.0f);
  success_tint_.TryUpdateTarget(active && success_ ? 1.0f : 0.0f);
  error_tint_.TryUpdateTarget(active && !success_ ? 1.0f : 0.0f);
  check_transition_.Update(delta_time / 0.42f);
  success_tint_.Update(delta_time / 0.30f);
  error_tint_.Update(delta_time / 0.30f);
  shake_phase_ += delta_time * 36.0f;
}

void FileButton::Draw() {
  const glm::vec2 position{left_, top_}, size{right_ - left_, bottom_ - top_};
  const float wave = std::sin(glm::pi<float>() * motion_);
  const float pulse = wave * wave;
  auto background = background_color_.GetValue(float(hover_));
  background = glm::mix(background, glm::vec4{0.16f, 0.25f, 0.20f, 1}, 0.65f * float(success_tint_));
  background = glm::mix(background, glm::vec4{0.30f, 0.13f, 0.13f, 1}, 0.65f * float(error_tint_));
  application_->DrawModel(background_, {GetModelMatrix(position, size, 0.6f), background, glm::uvec4{1, 0, 0, 0}});
  const float shake = std::sin(shake_phase_) * float(error_tint_) * 0.008f;
  auto icon_position = position + glm::vec2{shake * size.x, 0};
  static const MixValue<float> brightness({1.0f, 1.08f, 0.90f});
  auto color = button_palette::kNeutral * brightness.GetValue(float(hover_));
  application_->DrawModel(frame_.get(), {GetModelMatrix(icon_position, size, 0.4f), color, glm::uvec4{1, 0, 0, 0}});
  // Collapse and expand through zero instead of swapping full-size silhouettes.
  // Smoothstep keeps both ends and the midpoint at zero scale velocity.
  const float transition = float(check_transition_);
  const float check_scale = glm::smoothstep(0.5f, 1.0f, transition);
  const float arrow_scale = 1.0f - glm::smoothstep(0.0f, 0.5f, transition);
  if (check_scale > 0.0f)
    application_->DrawModel(
        check_.get(), {GetModelMatrix(icon_position + size * (1.0f - check_scale) * 0.5f, size * check_scale, 0.38f),
                       button_palette::kPlay, glm::uvec4{1, 0, 0, 0}});
  if (arrow_scale > 0.0f) {
    const float direction = kind_ == Kind::kSave ? 1.0f : -1.0f;
    icon_position.y += direction * pulse * size.y * 0.07f;
    application_->DrawModel(
        arrow_.get(), {GetModelMatrix(icon_position + size * (1.0f - arrow_scale) * 0.5f, size * arrow_scale, 0.38f),
                       color, glm::uvec4{1, 0, 0, 0}});
  }
}

void FileButton::Feedback(bool success) {
  success_ = success;
  feedback_ = success ? 1.2f : 0.65f;
}

void FileButton::BeginAction() {
  // Repeated activation must preserve the currently visible position.
  if (motion_ >= 1.0f)
    motion_ = 0.0f;
  feedback_ = 0.0f;
}

void FileButton::OnClick() {
  on_click_();
}

void FileButton::OnStateChange(int state) {
  hover_.UpdateTarget(float(state));
}
