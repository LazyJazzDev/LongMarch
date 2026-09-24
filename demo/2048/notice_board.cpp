#include "notice_board.h"

#include "application/application.h"
#include "rounded_rectangle.h"

NoticeBoard::NoticeBoard(Application *app,
                         font::Factory *font_factory,
                         const glm::vec3 &title_color,
                         const glm::vec3 &content_color,
                         const glm::vec3 &background_color,
                         const std::wstring &title_text,
                         const std::wstring &content_text)
    : app_(app) {
  right_ = 1.0f;
  bottom_ = 1.0f;
  title_text_ = title_text;
  content_text_ = content_text;
  background_color_ = background_color;
  title_bar_ = std::make_unique<TextBar>(app, font_factory, title_text, 1.0f, title_color, glm::vec2{0.0f, 0.0f},
                                         TextBar::AlignMode::kMid);
  content_bar_ = std::make_unique<TextBar>(app, font_factory, content_text, 1.0f, content_color, glm::vec2{0.0f, 0.0f},
                                           TextBar::AlignMode::kMid);
  auto model = GenerateRoundedRectangle(0.0f, 0.0f, 1.0f, 1.0f, 0.1f, background_color, 8);
  background_model_ = std::make_unique<DeviceModel>(app, model);
}

void NoticeBoard::Resize(float left, float top, float right, float bottom, float arc_radius) {
  left_ = left;
  top_ = top;
  right_ = right;
  bottom_ = bottom;
  arc_radius_ = arc_radius;
  if (top > bottom) {
    std::swap(top, bottom);
  }
  float separate_scale = 0.618;
  float separate_line = top + (bottom - top) * (1.0f - separate_scale);
  title_bar_->Resize((separate_line - top) * 0.618f,
                     glm::vec2{(left + right) * 0.5f, separate_line + (top - separate_line) * (1.0f - 0.618f) * 0.8f});
  float font_height = (bottom - separate_line) * 0.8f;
  content_font_size_ = font_height;
  UpdateContentText(content_text_);
  auto model = GenerateRoundedRectangle(left, top, right, bottom, arc_radius, background_color_, 8);
  background_model_ = std::make_unique<DeviceModel>(app_, model);
}

void NoticeBoard::Draw() {
  app_->DrawModel(background_model_.get(), InstanceInfo{glm::translate(glm::mat4{1.0f}, glm::vec3{0.0f, 0.0f, 0.8f}),
                                                        glm::vec4{1.0f}, glm::uvec4{0}});
  title_bar_->Draw();
  content_bar_->Draw();
}

void NoticeBoard::UpdateTitleText(const std::wstring &title_text) {
  title_text_ = title_text;
  title_bar_->UpdateText(title_text);
  // The title keeps its own layout: the bar is sized from the board rectangle,
  // not from the string, so a shorter title stays in the same place.
  float separate_scale = 0.618;
  float separate_line = top_ + (bottom_ - top_) * (1.0f - separate_scale);
  title_bar_->Resize(
      (separate_line - top_) * 0.618f,
      glm::vec2{(left_ + right_) * 0.5f, separate_line + (top_ - separate_line) * (1.0f - 0.618f) * 0.8f});
}

void NoticeBoard::UpdateTitleColor(const glm::vec3 &title_color) {
  title_bar_->UpdateColor(title_color);
}

void NoticeBoard::UpdateBackgroundColor(const glm::vec3 &background_color) {
  background_color_ = background_color;
  auto model = GenerateRoundedRectangle(left_, top_, right_, bottom_, arc_radius_, background_color_, 8);
  background_model_ = std::make_unique<DeviceModel>(app_, model);
}

void NoticeBoard::UpdateContentText(const std::wstring &content_text) {
  content_text_ = content_text;
  content_bar_->UpdateText(content_text);
  float font_height = GetFittingContentSize();
  float separate_scale = 0.618;
  float separate_line = top_ + (bottom_ - top_) * (1.0f - separate_scale);
  content_bar_->Resize(font_height,
                       glm::vec2{(left_ + right_) * 0.5f, (bottom_ + separate_line) * 0.5f + font_height * 0.3f});
}

float NoticeBoard::GetFittingContentSize() {
  const auto mesh = content_bar_->GetFontFactory()->GetString(content_text_);
  // TextBar centers normalized geometry around half the advance. Include any
  // glyph overhang so the rendered outline also stays inside the padded width.
  float width = mesh.GetAdvance();
  for (const auto &vertex : mesh.GetVertices())
    width = std::max(width, 2.0f * std::abs(vertex.x - mesh.GetAdvance() * 0.5f));
  if (width <= 0.0f)
    return content_font_size_;
  const float available_width = std::max(0.0f, right_ - left_) * 0.9f;
  return std::min(content_font_size_, available_width / width);
}
