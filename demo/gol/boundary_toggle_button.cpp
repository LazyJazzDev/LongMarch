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
  glider_.Update(dt);
  hover_.Update(dt * 10.0f);
}

void BoundaryToggleButton::Draw() {
  const glm::vec2 origin{left_, top_}, size{right_ - left_, bottom_ - top_};
  application_->DrawModel(background_, {GetModelMatrix(origin, size, 0.6f),
                                        button_palette::Background().GetValue(float(hover_)), glm::uvec4{1, 0, 0, 0}});
  application_->DrawModel(device_icon_.get(),
                          {GetModelMatrix(origin, size, 0.4f), glm::vec4{1.0f}, glm::uvec4{1, 0, 0, 0}});
  const auto cells = glider_.ProjectedCells();
  const glm::vec2 center = origin + size * 0.5f;
  constexpr float pitch = 0.20f;
  constexpr float half_size = 0.07f;
  // Center the initial 3x3 silhouette; the fourth row/column straddles the
  // periodic seam. Copies are clipped so wrapping cells remain full-sized.
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 4; ++x) {
      if (!cells[y * 4 + x])
        continue;
      for (int copy_y = -1; copy_y <= 0; ++copy_y) {
        for (int copy_x = -1; copy_x <= 0; ++copy_x) {
          const glm::vec2 p{(x - 1 + copy_x * 4) * pitch, (y - 1 + copy_y * 4) * pitch};
          const auto lo = glm::max(p - glm::vec2{half_size}, glm::vec2{-0.40f});
          const auto hi = glm::min(p + glm::vec2{half_size}, glm::vec2{0.40f});
          if (lo.x >= hi.x || lo.y >= hi.y)
            continue;
          application_->DrawModel(cell_model_.get(),
                                  {GetModelMatrix(center + lo * size * 0.5f, (hi - lo) * size * 0.5f, 0.4f),
                                   button_palette::kNeutral, glm::uvec4{1, 0, 0, 0}});
        }
      }
    }
  }
}

void BoundaryToggleButton::OnClick() {
  mode_ = mode_ == BoundaryMode::kPeriodic ? BoundaryMode::kFixed : BoundaryMode::kPeriodic;
  if (mode_ == BoundaryMode::kPeriodic)
    glider_.Start();
  else
    glider_.Reset();
}

void BoundaryToggleButton::OnStateChange(int state) {
  hover_.UpdateTarget(float(state));
}
