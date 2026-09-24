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
  // The cells stay neutral. Opposite portal edges share a color without
  // arrows or bridges suggesting a preferred direction of travel.
  for (float y : {-0.22f, 0.06f})
    for (float x : {-0.22f, 0.06f})
      rect(x, y, x + 0.16f, y + 0.16f, button_palette::kNeutral);
  const glm::vec4 horizontal = periodic ? glm::vec4{0.58f, 0.46f, 0.30f, 1.0f} : button_palette::kNeutral;
  const glm::vec4 vertical = periodic ? glm::vec4{0.32f, 0.50f, 0.62f, 1.0f} : button_palette::kNeutral;
  const float thickness = periodic ? 0.10f : 0.20f;
  // Small corner gaps distinguish the two portal pairs. In fixed mode the
  // same four quads expand into a continuous wall, keeping the spring morph.
  const float horizontal_extent = periodic ? 0.48f : 0.62f;
  const float vertical_extent = periodic ? 0.48f : 0.42f;
  for (float side : {-1.0f, 1.0f}) {
    const float low = side < 0 ? -0.62f : 0.62f - thickness;
    const float high = low + thickness;
    rect(-horizontal_extent, low, horizontal_extent, high, horizontal);
    rect(low, -vertical_extent, high, vertical_extent, vertical);
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
