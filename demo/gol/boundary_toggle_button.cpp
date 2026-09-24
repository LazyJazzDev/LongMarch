#include "boundary_toggle_button.h"

#include <algorithm>
#include <cmath>

#include "button_palette.h"

namespace {
Model BoundaryIcon(bool periodic) {
  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;
  auto triangle = [&](glm::vec2 a, glm::vec2 b, glm::vec2 c, glm::vec4 color) {
    const uint32_t start = vertices.size();
    auto v = ComposeVertices({a, b, c}, color);
    vertices.insert(vertices.end(), v.begin(), v.end());
    indices.insert(indices.end(), {start, start + 1, start + 2});
  };
  auto rect = [&](float left, float top, float right, float bottom, glm::vec4 color) {
    triangle({left, top}, {right, top}, {right, bottom}, color);
    triangle({left, top}, {right, bottom}, {left, bottom}, color);
  };
  // A stable, irregular 3x3 sample makes the repeated neighbors recognizable.
  // Keep the sample unchanged while toggling so only the boundary semantics move.
  constexpr bool cells[3][3] = {{true, false, true}, {false, true, false}, {true, true, false}};
  constexpr float pitch = 0.28f;
  constexpr float radius = 0.075f;
  for (int y = -1; y <= 3; ++y) {
    for (int x = -1; x <= 3; ++x) {
      if (!cells[(y + 3) % 3][(x + 3) % 3])
        continue;
      const bool outside = x < 0 || x > 2 || y < 0 || y > 2;
      const glm::vec2 center{(x - 1) * pitch, (y - 1) * pitch};
      const float half_size = outside && !periodic ? 0.0f : radius;
      const auto color =
          outside ? glm::vec4{glm::vec3(button_palette::kNeutral) * 0.5f, 1.0f} : button_palette::kNeutral;
      rect(center.x - half_size, center.y - half_size, center.x + half_size, center.y + half_size, color);
    }
  }
  // Contiguous wall segments shrink inward and separate into fine dashes.
  // Both states retain identical topology for the reversible spring morph.
  const float outer = periodic ? 0.425f : 0.62f;
  const float thickness = periodic ? 0.055f : 0.20f;
  const float gap = periodic ? 0.04f : 0.0f;
  for (float side : {-1.0f, 1.0f}) {
    const float low = side < 0 ? -outer : outer - thickness;
    for (int segment = 0; segment < 4; ++segment) {
      const float start = -outer + segment * outer * 0.5f + gap;
      const float end = -outer + (segment + 1) * outer * 0.5f - gap;
      rect(start, low, end, low + thickness, button_palette::kNeutral);
      rect(low, start, low + thickness, end, button_palette::kNeutral);
    }
  }
  return Model(vertices, indices);
}
}  // namespace

BoundaryToggleButton::BoundaryToggleButton(Application *app, DeviceModel *background)
    : Button(app, 0, 0, 1, 1),
      background_(background) {
  auto fixed = BoundaryIcon(false);
  auto periodic = BoundaryIcon(true);
  icon_ = std::make_unique<MixModel>(std::vector<std::vector<Vertex>>{fixed.Vertices(), periodic.Vertices()},
                                     fixed.Indices());
  device_icon_ = std::make_unique<DeviceModel>(app, icon_->GetModel(1.0f, MixStyle::kLinear));
}

void BoundaryToggleButton::Update(float seconds) {
  const float target = mode_ == BoundaryMode::kPeriodic ? 1.0f : 0.0f;
  const float dt = std::max(seconds, 0.0f);
  const float offset = morph_ - target;
  const float impulse = velocity_ + 24.0f * offset;
  const float decay = std::exp(-24.0f * dt);
  morph_ = target + (offset + impulse * dt) * decay;
  velocity_ = (velocity_ - 24.0f * impulse * dt) * decay;
  device_icon_->UploadVertices(icon_->GetModel(std::clamp(morph_, 0.0f, 1.0f), MixStyle::kLinear).Vertices());
  hover_.Update(dt * 10.0f);
}

void BoundaryToggleButton::Draw() {
  const glm::vec2 origin{left_, top_}, size{right_ - left_, bottom_ - top_};
  application_->DrawModel(background_, {GetModelMatrix(origin, size, 0.6f),
                                        button_palette::Background().GetValue(float(hover_)), glm::uvec4{1, 0, 0, 0}});
  application_->DrawModel(device_icon_.get(),
                          {GetModelMatrix(origin, size, 0.4f), glm::vec4{1.0f}, glm::uvec4{1, 0, 0, 0}});
}

void BoundaryToggleButton::OnClick() {
  mode_ = mode_ == BoundaryMode::kPeriodic ? BoundaryMode::kFixed : BoundaryMode::kPeriodic;
}

void BoundaryToggleButton::OnStateChange(int state) {
  hover_.UpdateTarget(float(state));
}
