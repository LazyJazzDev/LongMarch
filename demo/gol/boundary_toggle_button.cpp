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
  // The solid enclosure opens four portals without arrows or extra marks.
  const float gap = periodic ? 0.30f : 0.0f;
  for (float side : {-1.0f, 1.0f}) {
    const float low = side < 0 ? -0.62f : 0.42f;
    rect(-0.62f, low, -gap, low + 0.20f, button_palette::kNeutral);
    rect(gap, low, 0.62f, low + 0.20f, button_palette::kNeutral);
    rect(low, -0.42f, low + 0.20f, -gap, button_palette::kNeutral);
    rect(low, gap, low + 0.20f, 0.42f, button_palette::kNeutral);
  }
  return Model(vertices, indices);
}
}  // namespace

BoundaryToggleButton::BoundaryToggleButton(Application *app, DeviceModel *background)
    : Button(app, 0, 0, 1, 1),
      background_(background) {
  cell_model_ = std::make_unique<DeviceModel>(
      app, Model(ComposeVertices({{-1, -1}, {1, -1}, {1, 1}, {-1, 1}}, glm::vec4{1.0f}), {0, 1, 2, 0, 2, 3}));
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
  if (demo_active_) {
    demo_time_ += dt;
    if (demo_time_ >= 2.8f) {
      demo_active_ = false;
      demo_time_ = 0.0f;
    }
  }
  hover_.Update(dt * 10.0f);
}

void BoundaryToggleButton::Draw() {
  const glm::vec2 origin{left_, top_}, size{right_ - left_, bottom_ - top_};
  application_->DrawModel(background_, {GetModelMatrix(origin, size, 0.6f),
                                        button_palette::Background().GetValue(float(hover_)), glm::uvec4{1, 0, 0, 0}});
  application_->DrawModel(device_icon_.get(),
                          {GetModelMatrix(origin, size, 0.4f), glm::vec4{1.0f}, glm::uvec4{1, 0, 0, 0}});
  // A 4x4-sized torus shows the five-cell silhouette crossing opposite edges.
  // This is an explanatory translation, not a second Life simulation: actual
  // evolution on such a tiny torus would interfere with the glider itself.
  constexpr glm::vec2 cells[] = {{0, -1}, {1, 0}, {-1, 1}, {0, 1}, {1, 1}};
  constexpr float extent = 0.40f;
  constexpr float period = extent * 2.0f;
  const float progress = demo_active_ ? std::clamp((demo_time_ - 0.3f) / 2.5f, 0.0f, 1.0f) : 0.0f;
  const float travel = progress * progress * (3.0f - 2.0f * progress) * period;
  const glm::vec2 center = origin + size * 0.5f;
  for (const auto cell : cells) {
    for (int y = -1; y <= 1; ++y) {
      for (int x = -1; x <= 1; ++x) {
        const auto p = cell * 0.20f + glm::vec2{travel} + glm::vec2{x, y} * period;
        const auto lo = glm::max(p - glm::vec2{0.075f}, glm::vec2{-extent});
        const auto hi = glm::min(p + glm::vec2{0.075f}, glm::vec2{extent});
        if (lo.x >= hi.x || lo.y >= hi.y)
          continue;
        application_->DrawModel(cell_model_.get(),
                                {GetModelMatrix(center + lo * size * 0.5f, (hi - lo) * size * 0.5f, 0.4f),
                                 button_palette::kNeutral, glm::uvec4{1, 0, 0, 0}});
      }
    }
  }
}

void BoundaryToggleButton::OnClick() {
  mode_ = mode_ == BoundaryMode::kPeriodic ? BoundaryMode::kFixed : BoundaryMode::kPeriodic;
  demo_active_ = mode_ == BoundaryMode::kPeriodic;
  demo_time_ = 0.0f;
}

void BoundaryToggleButton::OnStateChange(int state) {
  hover_.UpdateTarget(float(state));
}
