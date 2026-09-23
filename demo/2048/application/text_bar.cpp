#include "text_bar.h"

#include <map>

TextBar::TextBar(Application *app,
                 font::Factory *font_factory,
                 const std::wstring &text,
                 float font_size,
                 glm::vec3 font_color,
                 glm::vec2 origin,
                 AlignMode align_mode)
    : Listener(app) {
  app_ = app;
  font_factory_ = font_factory;
  font_color_ = font_color;
  font_size_ = font_size;
  origin_ = origin;
  align_mode_ = align_mode;
  UpdateText(text);
}

TextBar::~TextBar() = default;

void TextBar::UpdateText(const std::wstring &text) {
  text_ = text;
  BuildMesh();
}

void TextBar::UpdateColor(const glm::vec3 &color) {
  font_color_ = color;
  BuildMesh();
}

void TextBar::BuildMesh() {
  auto triangles = font_factory_->GetString(text_);
  switch (align_mode_) {
    case AlignMode::kLeft:
      break;
    case AlignMode::kMid:
      for (auto &triangle : triangles.GetVertices()) {
        triangle.x -= triangles.GetAdvance() * 0.5f;
      }
      break;
    case AlignMode::kRight:
      for (auto &triangle : triangles.GetVertices()) {
        triangle.x -= triangles.GetAdvance();
      }
      break;
  }

  auto model = Model(ComposeVertices(triangles.GetVertices(), glm::vec4{1.0f}), triangles.GetIndices());
  device_model_ = std::make_unique<DeviceModel>(app_, model);
}

void TextBar::Draw() {
  application_->DrawModel(device_model_.get(),
                          InstanceInfo{glm::translate(glm::mat4{1.0f}, glm::vec3{origin_, 0.0f}) *
                                           glm::scale(glm::mat4{1.0f}, glm::vec3{font_size_, -font_size_, 1.0f}),
                                       glm::vec4{font_color_, 1.0f}, glm::uvec4{0, 0, 0, 0}});
}

void TextBar::Resize(float font_size, glm::vec2 origin) {
  font_size_ = font_size;
  origin_ = origin;
}

font::Factory *TextBar::GetFontFactory() {
  return font_factory_;
}
