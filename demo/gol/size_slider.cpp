#include "size_slider.h"

#include <array>
#include <string>

#include "grid_size.h"

SizeSlider::SizeSlider(Application *app,
                       char label,
                       int value,
                       DeviceModel *rectangle,
                       std::function<void(int)> on_change)
    : Listener(app),
      label_(label),
      value_(value),
      rectangle_(rectangle),
      on_change_(std::move(on_change)) {
  RebuildLabel();
  key_callback_ = app->GetWindow()->KeyEvent().RegisterCallback([this](int key, int, int action, int mods) {
    if (!focused_ || (action != GLFW_PRESS && action != GLFW_REPEAT) ||
        (mods & (GLFW_MOD_CONTROL | GLFW_MOD_SUPER | GLFW_MOD_ALT)))
      return;
    if (key == GLFW_KEY_LEFT || key == GLFW_KEY_DOWN)
      SetValue(value_ - 1);
    else if (key == GLFW_KEY_RIGHT || key == GLFW_KEY_UP)
      SetValue(value_ + 1);
    else if (key == GLFW_KEY_HOME)
      SetValue(grid_size::kMin);
    else if (key == GLFW_KEY_END)
      SetValue(grid_size::kMax);
  });
}

SizeSlider::~SizeSlider() {
  application_->GetWindow()->KeyEvent().UnregisterCallback(key_callback_);
  application_->UnregisterListener(this);
}

void SizeSlider::Resize(glm::vec4 bounds, bool vertical) {
  bounds_ = bounds;
  vertical_ = vertical;
}

glm::vec2 SizeSlider::FramePosition(double x, double y) const {
  const auto size = application_->GetWindow()->GetSize();
  return glm::vec2{float(x), float(y)} * glm::vec2(application_->FramebufferSize()) /
         glm::vec2(glm::max(size, glm::ivec2{1}));
}

bool SizeSlider::Contains(glm::vec2 p) const {
  return p.x >= bounds_.x && p.x <= bounds_.z && p.y >= bounds_.y && p.y <= bounds_.w;
}

void SizeSlider::OnMouseButton(int button, int action, int) {
  if (button != GLFW_MOUSE_BUTTON_LEFT)
    return;
  const auto cursor = application_->GetWindow()->GetCursorPosition();
  const auto p = FramePosition(cursor.x, cursor.y);
  if (action == GLFW_PRESS) {
    focused_ = Contains(p);
    dragging_ = focused_;
    if (dragging_)
      DragTo(p);
  } else if (action == GLFW_RELEASE) {
    if (dragging_)
      DragTo(p);
    dragging_ = false;
  }
}

void SizeSlider::OnCursorPos(double x, double y) {
  const auto p = FramePosition(x, y);
  hovered_ = Contains(p);
  if (dragging_)
    DragTo(p);
}

void SizeSlider::OnCursorEnter(int entered) {
  if (!entered)
    hovered_ = false;
}

void SizeSlider::OnFocus(bool focused) {
  if (!focused) {
    dragging_ = false;
    hovered_ = false;
    focused_ = false;
  }
}

void SizeSlider::DragTo(glm::vec2 p) {
  const float length = vertical_ ? bounds_.w - bounds_.y : bounds_.z - bounds_.x;
  const float position = vertical_ ? bounds_.w - p.y : p.x - bounds_.x;
  if (length > 0.0f)
    SetValue(grid_size::FromFraction(position / length));
}

void SizeSlider::SetValue(int value) {
  value = std::clamp(value, grid_size::kMin, grid_size::kMax);
  if (value == value_)
    return;
  value_ = value;
  RebuildLabel();
  on_change_(value_);
}

void SizeSlider::RebuildLabel() {
  // Normal-width glyphs with bold, uniform strokes and small rounded corners.
  using Path = std::vector<glm::vec2>;
  static const std::array<std::vector<Path>, 12> glyphs{
      {{{{0, 0}, {6, 0}, {6, 6}, {0, 6}, {0, 0}}},
       {{{1.5f, 1.5f}, {3, 0}, {3, 6}}, {{0, 6}, {6, 6}}},
       {{{0, 0}, {6, 0}, {6, 3}, {0, 3}, {0, 6}, {6, 6}}},
       {{{0, 0}, {6, 0}, {6, 6}, {0, 6}}, {{1.5f, 3}, {6, 3}}},
       {{{0, 0}, {0, 3}, {6, 3}}, {{6, 0}, {6, 6}}},
       {{{6, 0}, {0, 0}, {0, 3}, {6, 3}, {6, 6}, {0, 6}}},
       {{{6, 0}, {0, 0}, {0, 6}, {6, 6}, {6, 3}, {0, 3}}},
       {{{0, 0}, {6, 0}, {6, 6}}},
       {{{0, 0}, {6, 0}, {6, 6}, {0, 6}, {0, 0}}, {{0, 3}, {6, 3}}},
       {{{6, 3}, {0, 3}, {0, 0}, {6, 0}, {6, 6}, {0, 6}}},
       {{{0, 0}, {0, 6}, {3, 6}, {3, 2.5f}}, {{3, 6}, {6, 6}, {6, 0}}},
       {{{0, 0}, {0, 6}}, {{6, 0}, {6, 6}}, {{0, 3}, {6, 3}}}}};
  const std::string text = std::string(1, label_) + " " + std::to_string(value_);
  constexpr float advance = 6.2f;
  label_width_ = float(text.size() - 1) * advance + 4.0f;
  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;
  auto capsule = [&](glm::vec2 from, glm::vec2 to) {
    constexpr float radius = 0.95f;
    constexpr int steps = 10;
    const float angle = std::atan2(to.y - from.y, to.x - from.x);
    const uint32_t base = uint32_t(vertices.size());
    for (int cap = 0; cap < 2; ++cap) {
      const auto center = cap == 0 ? to : from;
      for (int j = 0; j <= steps; ++j) {
        const float theta = angle - glm::half_pi<float>() + (float(cap) + float(j) / steps) * glm::pi<float>();
        vertices.push_back({center + radius * glm::vec2{std::cos(theta), std::sin(theta)}, glm::vec4{1.0f}});
      }
    }
    for (uint32_t j = 1; j + 1 < 2 * (steps + 1); ++j)
      for (auto index : {base, base + j, base + j + 1})
        indices.push_back(index);
  };
  for (size_t i = 0; i < text.size(); ++i) {
    if (text[i] == ' ')
      continue;
    const int glyph = text[i] == 'W' ? 10 : text[i] == 'H' ? 11 : text[i] - '0';
    for (auto path : glyphs[glyph]) {
      // Narrow the centerlines before stroking, so vertical strokes retain their weight.
      for (auto &point : path)
        point.x *= 2.0f / 3.0f;
      const bool closed = path.front() == path.back();
      const size_t count = path.size() - (closed ? 1 : 0);
      Path rounded;
      for (size_t j = 0; j < count; ++j) {
        if (!closed && (j == 0 || j + 1 == count)) {
          rounded.push_back(path[j]);
          continue;
        }
        const auto before = path[(j + count - 1) % count] - path[j];
        const auto after = path[(j + 1) % count] - path[j];
        const float trim = std::min(0.65f, std::min(glm::length(before), glm::length(after)) * 0.4f);
        const auto start = path[j] + glm::normalize(before) * trim;
        const auto end = path[j] + glm::normalize(after) * trim;
        for (int sample = 0; sample <= 6; ++sample) {
          const float t = float(sample) / 6.0f;
          rounded.push_back(glm::mix(glm::mix(start, path[j], t), glm::mix(path[j], end, t), t));
        }
      }
      if (closed)
        rounded.push_back(rounded.front());
      for (size_t j = 1; j < rounded.size(); ++j)
        capsule(rounded[j - 1] + glm::vec2{float(i) * advance, 0}, rounded[j] + glm::vec2{float(i) * advance, 0});
    }
  }
  if (!label_model_)
    label_model_ = std::make_unique<DeviceModel>(application_, Model(vertices, indices));
  else {
    label_model_->UploadVertices(vertices);
    label_model_->UploadIndices(indices);
  }
}

void SizeSlider::RoundedRect(glm::vec2 position,
                             glm::vec2 size,
                             float radius,
                             float depth,
                             glm::vec4 color,
                             glm::vec4 clip,
                             bool recessed) {
  application_->DrawModel(rectangle_, {GetModelMatrix(position, size, depth),
                                       color,
                                       {recessed ? 5u : 4u, glm::floatBitsToUint(size.x * 0.5f),
                                        glm::floatBitsToUint(size.y * 0.5f), glm::floatBitsToUint(radius)},
                                       clip});
}

void SizeSlider::Draw() {
  const glm::vec2 position{bounds_.x, bounds_.y};
  const glm::vec2 size{bounds_.z - bounds_.x, bounds_.w - bounds_.y};
  const float thickness = vertical_ ? size.x : size.y;
  const float length = vertical_ ? size.y : size.x;
  const float radius = thickness * 0.18f;
  const float fraction = float(value_ - grid_size::kMin) / float(grid_size::kMax - grid_size::kMin);
  const float highlight = dragging_ || hovered_ || focused_ ? 0.035f : 0.0f;
  RoundedRect(position, size, radius, 0.5f, {0.20f + highlight, 0.23f + highlight, 0.28f + highlight, 1.0f}, bounds_,
              true);
  // Clip a second copy of the same rounded silhouette at the value boundary.
  // This produces a straight color division without a separate handle or seam.
  auto filled = bounds_;
  if (vertical_)
    filled.y = bounds_.w - size.y * fraction;
  else
    filled.z = bounds_.x + size.x * fraction;
  if (fraction > 0.0f)
    RoundedRect(position, size, radius, 0.45f, {0.34f + highlight, 0.42f + highlight, 0.53f + highlight, 1.0f}, filled,
                false);

  const float pixel = std::min(thickness * 0.34f / 7.9f, length * 0.8f / label_width_);
  auto transform = glm::translate(glm::mat4{1.0f}, glm::vec3{position + size * 0.5f, 0.4f}) *
                   glm::rotate(glm::mat4{1.0f}, vertical_ ? -glm::half_pi<float>() : 0.0f, glm::vec3{0, 0, 1}) *
                   glm::scale(glm::mat4{1.0f}, glm::vec3{pixel, pixel, 1.0f}) *
                   glm::translate(glm::mat4{1.0f}, glm::vec3{-label_width_ * 0.5f, -3.0f, 0});
  application_->DrawModel(label_model_.get(), {transform, {0.55f, 0.60f, 0.67f, 1.0f}, glm::uvec4{0}, bounds_});
}
