#include "boundary_toggle_button.h"

#include <algorithm>
#include <cmath>

#include "button_palette.h"

namespace {
Model BoundaryIcon(bool periodic) {
  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;
  const glm::vec4 color = periodic ? glm::vec4{0.40f, 0.53f, 0.54f, 1.0f} : button_palette::kNeutral;
  auto point = [periodic](float angle) {
    glm::vec2 p{std::cos(angle), std::sin(angle)};
    return p * (periodic ? 0.52f : 0.48f / std::max(std::abs(p.x), std::abs(p.y)));
  };
  auto triangle = [&](glm::vec2 a, glm::vec2 b, glm::vec2 c) {
    const uint32_t start = vertices.size();
    auto v = ComposeVertices({a, b, c}, color);
    vertices.insert(vertices.end(), v.begin(), v.end());
    indices.insert(indices.end(), {start, start + 1, start + 2});
  };
  // The square enclosure opens into two curved arrows, retaining mesh topology
  // so rapid toggles can reverse the same continuous morph.
  for (int half = 0; half < 2; ++half) {
    const float start = half * glm::pi<float>() + (periodic ? 0.22f : 0.0f);
    const float end = (half + 1) * glm::pi<float>() - (periodic ? 0.22f : 0.0f);
    for (int i = 0; i < 24; ++i) {
      const auto a = point(glm::mix(start, end, i / 24.0f));
      const auto b = point(glm::mix(start, end, (i + 1) / 24.0f));
      triangle(a * 0.87f, a * 1.13f, b * 1.13f);
      triangle(a * 0.87f, b * 1.13f, b * 0.87f);
    }
    const auto tip = point(end);
    const glm::vec2 tangent{-std::sin(end), std::cos(end)};
    const glm::vec2 radial{std::cos(end), std::sin(end)};
    const float size = periodic ? 1.0f : 0.0f;
    triangle(tip + tangent * (0.19f * size), tip - tangent * (0.08f * size) + radial * (0.18f * size),
             tip - tangent * (0.08f * size) - radial * (0.18f * size));
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
