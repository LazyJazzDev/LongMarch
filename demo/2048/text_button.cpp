#include "text_button.h"

#include <utility>

TextButton::TextButton(Application *app,
                       font::Factory *font_factory,
                       glm::vec3 foreground_color,
                       glm::vec3 background_color,
                       const std::wstring &text,
                       std::function<void(Application *app)> callback_func,
                       float font_scale)
    : Button(app, 0.0f, 0.0f, 1.0f, 1.0f) {
  background_color_ = background_color;
  font_scale_ = font_scale;
  text_bar_ = std::make_unique<TextBar>(app, font_factory, text, 1.0f, foreground_color, glm::vec2{0.0f, 0.0f},
                                        TextBar::AlignMode::kMid);
  auto model = GenerateRoundedRectangle(0.0f, 0.0f, 1.0f, 1.0f, 0.1f, background_color);
  background_model_ = std::make_unique<DeviceModel>(app_, model);
  callback_func_ = std::move(callback_func);
}

void TextButton::Resize(float left, float top, float right, float bottom, float arc_size) {
  arc_size_ = arc_size;
  Button::Resize(left, top, right, bottom);
}

void TextButton::UpdateText(const std::wstring &text) {
  text_bar_->UpdateText(text);
}

void TextButton::UpdateBackgroundColor(glm::vec3 color) {
  background_color_ = color;
  OnResize();
}

void TextButton::Draw() {
  application_->DrawModel(
      background_model_.get(),
      InstanceInfo{glm::translate(glm::mat4{1.0f}, glm::vec3{0.0f, 0.0f, 0.8f}), glm::vec4{1.0f}, glm::uvec4{0}});
  text_bar_->Draw();
}

void TextButton::OnClick() {
  Button::OnClick();
  callback_func_(app_);
}

void TextButton::OnResize() {
  auto model = GenerateRoundedRectangle(left_, top_, right_, bottom_, arc_size_, background_color_);
  float font_size = (bottom_ - top_) * font_scale_;
  text_bar_->Resize(font_size, glm::vec2{(left_ + right_) * 0.5f,
                                         bottom_ - (bottom_ - top_ - font_size) * 0.5f - font_size * 0.125f});
  background_model_ = std::make_unique<DeviceModel>(app_, model);
}
