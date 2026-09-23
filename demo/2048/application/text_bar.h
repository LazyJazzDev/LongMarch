#pragma once

#include "font/factory.h"
#include "listener.h"
#include "model.h"

class TextBar : public Listener {
 public:
  enum class AlignMode : uint32_t { kLeft, kMid, kRight };

  TextBar(Application *app,
          font::Factory *font_factory,
          const std::wstring &text,
          float font_size,
          glm::vec3 font_color,
          glm::vec2 origin,
          AlignMode align_mode = AlignMode::kLeft);

  ~TextBar() override;

  void UpdateText(const std::wstring &text);

  void UpdateColor(const glm::vec3 &color);

  void Draw();

  void Resize(float font_size, glm::vec2 origin);

  font::Factory *GetFontFactory();

 private:
  void BuildMesh();

  Application *app_;
  font::Factory *font_factory_;
  float font_size_;
  glm::vec2 origin_{};
  AlignMode align_mode_;
  glm::vec3 font_color_{};
  std::unique_ptr<DeviceModel> device_model_;
  std::wstring text_;
};
