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
    const float low = side < 0 ? -0.62f : 0.46f;
    rect(-0.62f, low, -gap, low + 0.16f, button_palette::kNeutral);
    rect(gap, low, 0.62f, low + 0.16f, button_palette::kNeutral);
    rect(low, -0.46f, low + 0.16f, -gap, button_palette::kNeutral);
    rect(low, gap, low + 0.16f, 0.46f, button_palette::kNeutral);
  }
  return Model(vertices, indices);
}

// Cell vertices use button-local coordinates, just like the enclosure, so
// IconTheme evaluates one continuous light field across the complete icon.
Model GliderModel(const std::array<uint8_t, 16> &cells) {
  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;
  constexpr float pitch = 0.18f;
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 4; ++x) {
      const glm::vec2 p{(x - 1.5f) * pitch, (y - 1.5f) * pitch};
      const float radius = cells[y * 4 + x] ? pitch * 0.5f : 0.0f;
      const uint32_t base = vertices.size();
      auto quad = ComposeVertices({p + glm::vec2{-radius, -radius}, p + glm::vec2{radius, -radius},
                                   p + glm::vec2{radius, radius}, p + glm::vec2{-radius, radius}},
                                  glm::vec4{1.0f});
      vertices.insert(vertices.end(), quad.begin(), quad.end());
      indices.insert(indices.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
    }
  }
  return Model(vertices, indices);
}
}  // namespace

BoundaryToggleButton::BoundaryToggleButton(Application *app, DeviceModel *background)
    : Button(app, 0, 0, 1, 1),
      background_(background) {
  drawn_cells_ = glider_.ProjectedCells();
  cell_model_ = std::make_unique<DeviceModel>(app, GliderModel(drawn_cells_));
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
  const auto cells = glider_.ProjectedCells();
  if (cells != drawn_cells_) {
    cell_model_->UploadVertices(GliderModel(cells).Vertices());
    drawn_cells_ = cells;
  }
  hover_.Update(dt * 10.0f);
}

void BoundaryToggleButton::Draw() {
  const glm::vec2 origin{left_, top_}, size{right_ - left_, bottom_ - top_};
  application_->DrawModel(background_, {GetModelMatrix(origin, size, 0.6f),
                                        button_palette::Background().GetValue(float(hover_)), glm::uvec4{1, 0, 0, 0}});
  application_->DrawModel(device_icon_.get(),
                          {GetModelMatrix(origin, size, 0.4f), glm::vec4{1.0f}, glm::uvec4{1, 0, 0, 0}});
  application_->DrawModel(cell_model_.get(),
                          {GetModelMatrix(origin, size, 0.4f), button_palette::kNeutral, glm::uvec4{1, 0, 0, 0}});
}

void BoundaryToggleButton::OnClick() {
  mode_ = mode_ == BoundaryMode::kPeriodic ? BoundaryMode::kFixed : BoundaryMode::kPeriodic;
  glider_.Start(mode_ == BoundaryMode::kPeriodic);
}

void BoundaryToggleButton::OnStateChange(int state) {
  hover_.UpdateTarget(float(state));
}
