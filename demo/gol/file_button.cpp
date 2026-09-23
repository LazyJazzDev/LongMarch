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
