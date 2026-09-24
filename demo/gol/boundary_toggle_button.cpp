#include "boundary_toggle_button.h"

#include <algorithm>
#include <cmath>

#include "button_palette.h"

namespace {
Model BoundaryIcon(bool periodic) {
  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;
  const glm::vec4 color = periodic ? glm::vec4{0.40f, 0.53f, 0.54f, 1.0f} : button_palette::kNeutral;
  auto triangle = [&](glm::vec2 a, glm::vec2 b, glm::vec2 c) {
    const uint32_t start = vertices.size();
    auto v = ComposeVertices({a, b, c}, color);
    vertices.insert(vertices.end(), v.begin(), v.end());
    indices.insert(indices.end(), {start, start + 1, start + 2});
  };
  auto rect = [&](float left, float top, float right, float bottom) {
    triangle({left, top}, {right, top}, {right, bottom});
    triangle({left, top}, {right, bottom}, {left, bottom});
  };
  // Keep a small cell grid visible in both states. Thick walls open at their
  // centers, where symmetric double-headed links show adjacency across edges.
  for (float y : {-0.22f, 0.06f})
    for (float x : {-0.22f, 0.06f})
      rect(x, y, x + 0.16f, y + 0.16f);
  const float gap = periodic ? 0.22f : 0.0f;
  for (float side : {-1.0f, 1.0f}) {
    const float low = side < 0 ? -0.62f : 0.42f;
    const float high = low + 0.20f;
    rect(-0.62f, low, -gap, high);
    rect(gap, low, 0.62f, high);
    rect(low, -0.42f, high, -gap);
    rect(low, gap, high, 0.42f);
  }
  // Preserve the vertex topology for the spring morph; fixed-mode arrows
  // collapse into the wall centers instead of fading to disconnected shapes.
  for (int axis = 0; axis < 2; ++axis) {
    for (float side : {-1.0f, 1.0f}) {
      const glm::vec2 center = axis == 0 ? glm::vec2{side * 0.52f, 0} : glm::vec2{0, side * 0.52f};
      auto local = [&](float x, float y) {
        const glm::vec2 p = axis == 0 ? glm::vec2{x, y} : glm::vec2{y, x};
        return center + (periodic ? p : glm::vec2{0});
      };
      triangle(local(-0.10f, -0.08f), local(0.10f, -0.08f), local(0.10f, 0.08f));
      triangle(local(-0.10f, -0.08f), local(0.10f, 0.08f), local(-0.10f, 0.08f));
      triangle(local(0.07f, -0.17f), local(0.27f, 0), local(0.07f, 0.17f));
      triangle(local(-0.07f, 0.17f), local(-0.27f, 0), local(-0.07f, -0.17f));
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
