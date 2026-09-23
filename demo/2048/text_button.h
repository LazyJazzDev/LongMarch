#include <functional>

#include "application/button.h"
#include "application/text_bar.h"
#include "rounded_rectangle.h"

class TextButton : public Button {
 public:
  TextButton(Application *app,
             font::Factory *font_factory,
             glm::vec3 foreground_color,
             glm::vec3 background_color,
             const std::wstring &text,
             std::function<void(Application *app)> callback_func,
             float font_scale);
  void Resize(float left, float top, float right, float bottom, float arc_size);
  void UpdateText(const std::wstring &text);
  void UpdateBackgroundColor(glm::vec3 color);
  void Draw();
  void OnResize() override;
  void OnClick() override;

 private:
  float font_scale_{0.618f};
  std::function<void(Application *app)> callback_func_;
  float arc_size_{0.0f};
  glm::vec3 background_color_{0.0f};
  std::unique_ptr<DeviceModel> background_model_;
  std::unique_ptr<TextBar> text_bar_;
};
