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
  int w, h;
  glfwGetWindowSize(application_->GLFWWindow(), &w, &h);
  return glm::vec2{float(x), float(y)} * glm::vec2(application_->FramebufferSize()) /
         glm::vec2{std::max(w, 1), std::max(h, 1)};
}

bool SizeSlider::Contains(glm::vec2 p) const {
  return p.x >= bounds_.x && p.x <= bounds_.z && p.y >= bounds_.y && p.y <= bounds_.w;
}

void SizeSlider::OnMouseButton(int button, int action, int) {
  if (button != GLFW_MOUSE_BUTTON_LEFT)
    return;
  double x, y;
  glfwGetCursorPos(application_->GLFWWindow(), &x, &y);
  const auto p = FramePosition(x, y);
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
  // Compact 5x7 glyphs keep the small labels legible without a font texture.
  static constexpr std::array<std::array<uint8_t, 7>, 12> glyphs{{{{14, 17, 19, 21, 25, 17, 14}},
                                                                  {{4, 12, 4, 4, 4, 4, 14}},
                                                                  {{14, 17, 1, 2, 4, 8, 31}},
                                                                  {{30, 1, 1, 14, 1, 1, 30}},
                                                                  {{2, 6, 10, 18, 31, 2, 2}},
                                                                  {{31, 16, 16, 30, 1, 1, 30}},
                                                                  {{14, 16, 16, 30, 17, 17, 14}},
                                                                  {{31, 1, 2, 4, 8, 8, 8}},
                                                                  {{14, 17, 17, 14, 17, 17, 14}},
                                                                  {{14, 17, 17, 15, 1, 1, 14}},
                                                                  {{17, 17, 17, 21, 21, 27, 17}},
                                                                  {{17, 17, 17, 31, 17, 17, 17}}}};
  std::string text = std::string(1, label_) + " " + std::to_string(value_);
  label_width_ = float(text.size() * 6 - 1);
  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;
  for (size_t i = 0; i < text.size(); ++i) {
    if (text[i] == ' ')
      continue;
    int glyph = text[i] == 'W' ? 10 : text[i] == 'H' ? 11 : text[i] - '0';
    for (int y = 0; y < 7; ++y) {
      for (int x = 0; x < 5; ++x) {
        if (!(glyphs[glyph][y] & (1 << (4 - x))))
          continue;
        float left = float(i * 6 + x), top = float(y);
        uint32_t base = uint32_t(vertices.size());
        for (auto p :
             {glm::vec2{left, top}, glm::vec2{left + 1, top}, glm::vec2{left + 1, top + 1}, glm::vec2{left, top + 1}})
          vertices.push_back({p, glm::vec4{1.0f}});
        for (uint32_t index : {0u, 1u, 2u, 0u, 2u, 3u})
          indices.push_back(base + index);
      }
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
                             glm::vec4 clip) {
  application_->DrawModel(rectangle_, {GetModelMatrix(position, size, depth),
                                       color,
                                       {4u, glm::floatBitsToUint(size.x * 0.5f), glm::floatBitsToUint(size.y * 0.5f),
                                        glm::floatBitsToUint(radius)},
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
  RoundedRect(position, size, radius, 0.5f, {0.20f + highlight, 0.23f + highlight, 0.28f + highlight, 1.0f}, bounds_);
  // Clip a second copy of the same rounded silhouette at the value boundary.
  // This produces a straight color division without a separate handle or seam.
  auto filled = bounds_;
  if (vertical_)
    filled.y = bounds_.w - size.y * fraction;
  else
    filled.z = bounds_.x + size.x * fraction;
  if (fraction > 0.0f)
    RoundedRect(position, size, radius, 0.45f, {0.34f + highlight, 0.42f + highlight, 0.53f + highlight, 1.0f}, filled);

  const float pixel = std::min(thickness * 0.36f / 7.0f, length * 0.8f / label_width_);
  auto transform = glm::translate(glm::mat4{1.0f}, glm::vec3{position + size * 0.5f, 0.4f}) *
                   glm::rotate(glm::mat4{1.0f}, vertical_ ? -glm::half_pi<float>() : 0.0f, glm::vec3{0, 0, 1}) *
                   glm::scale(glm::mat4{1.0f}, glm::vec3{pixel, pixel, 1.0f}) *
                   glm::translate(glm::mat4{1.0f}, glm::vec3{-label_width_ * 0.5f, -3.5f, 0});
  application_->DrawModel(label_model_.get(), {transform, {0.55f, 0.60f, 0.67f, 1.0f}, glm::uvec4{0}, bounds_});
}
