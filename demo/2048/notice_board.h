#pragma once
#include "application/text_bar.h"

class NoticeBoard {
 public:
  NoticeBoard(Application *app,
              font::Factory *font_factory,
              const glm::vec3 &title_color,
              const glm::vec3 &content_color,
              const glm::vec3 &background_color,
              const std::wstring &title_text,
              const std::wstring &content_text);
  void Resize(float left, float top, float right, float bottom, float arc_radius);
  void Draw();
  void UpdateContentText(const std::wstring &content_text);

 private:
  Application *app_{};
  float GetFittingContentSize();
  float left_{0.0f};
  float top_{0.0f};
  float right_{0.0f};
  float bottom_{0.0f};
  float arc_radius_{0.0f};
  float content_font_size_{1.0f};
  std::unique_ptr<DeviceModel> background_model_;
  std::unique_ptr<TextBar> title_bar_;
  std::unique_ptr<TextBar> content_bar_;
  glm::vec3 background_color_{1.0f};
  std::wstring title_text_{L"TITLE"};
  std::wstring content_text_{L"CONTENT"};
};
