#pragma once

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "grassland/graphics/interface.h"

namespace snowberg::draw {
class Core;
class Model;
}  // namespace snowberg::draw

namespace snowberg::gui {
struct Color {
  float r{}, g{}, b{}, a{1.0f};
};

struct Theme {
  Color panel{0.93f, 0.96f, 1.0f, 0.68f};
  Color control{1.0f, 1.0f, 1.0f, 0.36f};
  Color accent{0.04f, 0.43f, 0.91f, 1.0f};
  Color text{0.08f, 0.12f, 0.19f, 1.0f};
  Color muted{0.29f, 0.36f, 0.45f, 1.0f};
  bool backdrop_blur{true};
  bool dark{false};
};

std::string DefaultFont();

class Context;

class Canvas {
 public:
  void Rectangle(float x, float y, float width, float height, Color color, float radius = 8.0f);
  void Circle(float x, float y, float radius, Color color);
  void Label(float x, float baseline, const std::string &text, Color color, unsigned size = 15);
  void ParticleField(int count, float time, Color color);

  float Width() const {
    return width_;
  }

  float Height() const {
    return height_;
  }

 private:
  friend class Context;

  Canvas(Context *context, float x, float y, float width, float height)
      : context_(context),
        x_(x),
        y_(y),
        width_(width),
        height_(height) {
  }

  Context *context_{};
  float x_{}, y_{}, width_{}, height_{};
};

struct CanvasResponse {
  glm::vec2 pointer{};
  bool hovered{}, pressed{}, clicked{};
};

// Declarative widgets with persistent pointer capture and stable string IDs.
// Rebuild the visible controls once per frame and render into the scene image.
class Context {
 public:
  Context(grassland::graphics::Core *core, grassland::graphics::Window *window, const std::string &font_file);
  ~Context();
  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;

  Theme &Style() {
    return theme_;
  }

  void SetPointerInput(float x, float y, bool down);
  void ClearPointerInput();
  void SetLogicalSize(int width, int height);
  void BeginFrame();
  void BeginPanel(const std::string &id, float x, float y, float width, const std::string &title = {});
  void EndPanel();
  void BeginRow(int columns);
  void EndRow();
  void Text(const std::string &text);
  void Heading(const std::string &text);
  void Spacer(float height = 8.0f);
  void Separator();
  bool Button(const std::string &label);
  bool Checkbox(const std::string &label, bool *value);
  bool Slider(const std::string &label, float *value, float min, float max);
  bool Slider(const std::string &label, int *value, int min, int max);
  bool Choice(const std::string &label, int *index, const std::vector<std::string> &items);
  void Plot(const std::string &label, const std::vector<float> &values, float min, float max);
  CanvasResponse Custom(const std::string &id, float height, std::function<void(Canvas &)> paint);
  // Returns the composited image. Keep it alive through command submission.
  grassland::graphics::Image *EndFrame(grassland::graphics::CommandContext *commands,
                                       grassland::graphics::Image *scene);

  bool WantsPointer() const {
    return wants_pointer_;
  }

 private:
  friend class Canvas;
  struct Item;
  struct Panel;

  struct Frame {
    float x{}, y{}, width{};
  };

  bool Hit(float x, float y, float width, float height) const;
  bool Activate(const std::string &id, float x, float y, float width, float height);
  float NextY(float height);
  Frame NextFrame(float height);
  void DrawRect(float x, float y, float width, float height, Color color, float radius = 12.0f);
  void DrawText(float x, float y, const std::string &text, Color color, unsigned size);
  void PaintItem(const Item &item);

  grassland::graphics::Core *core_{};
  grassland::graphics::Window *window_{};
  std::unique_ptr<draw::Core> painter_;
  std::map<std::tuple<int, int, int>, std::unique_ptr<draw::Model>> rect_models_;
  Theme theme_;
  std::vector<Panel> panels_;
  Panel *current_{};
  std::string active_id_;
  bool pointer_down_{}, pointer_pressed_{}, pointer_released_{}, wants_pointer_{};
  float cursor_x_{}, cursor_y_{}, scale_x_{1.0f}, scale_y_{1.0f};
  int target_width_{}, target_height_{};
  std::optional<glm::vec3> pointer_override_;
  glm::ivec2 logical_override_{0};
  std::unique_ptr<grassland::graphics::Shader> blur_shader_, composite_shader_;
  std::unique_ptr<grassland::graphics::ComputeProgram> blur_program_, composite_program_;
  std::unique_ptr<grassland::graphics::Buffer> blur_settings_horizontal_, blur_settings_vertical_, glass_settings_;
  std::unique_ptr<grassland::graphics::Image> horizontal_, vertical_, output_;
};
}  // namespace snowberg::gui
