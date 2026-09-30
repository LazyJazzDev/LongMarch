#include "snowberg/gui/gui.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "glm/gtc/matrix_transform.hpp"
#include "snowberg/draw/draw_core.h"
#include "snowberg/draw/draw_font.h"
#include "snowberg/draw/draw_model.h"

namespace snowberg::gui {
namespace {
#include "built_in_shaders.inl"

glm::vec4 Vec(Color c) {
  return {c.r, c.g, c.b, c.a};
}

Color Alpha(Color c, float a) {
  c.a *= a;
  return c;
}

struct BlurSettings {
  glm::ivec2 direction, extent;
};

struct GlassSettings {
  glm::ivec2 extent;
  int count{}, pad{};
  glm::vec4 rects[16]{};
};

std::string ValueLabel(const std::string &label, float value, float range) {
  std::ostringstream stream;
  stream << label << "  " << std::fixed << std::setprecision(range < 0.5f ? 3 : range < 10.0f ? 2 : 1) << value;
  return stream.str();
}
}  // namespace

std::string DefaultFont() {
#ifdef __APPLE__
  if (std::filesystem::exists("/System/Library/Fonts/SFNS.ttf"))
    return "/System/Library/Fonts/SFNS.ttf";
#endif
  return grassland::FindAssetFile("fonts/ClearSans-Bold-webfont.woff");
}

struct Context::Item {
  enum class Kind { kText, kHeading, kSeparator, kButton, kCheckbox, kSlider, kChoice, kPlot, kCanvas } kind;
  std::string label;
  float x{}, y{}, width{}, height{}, value{};
  bool checked{}, hovered{}, active{};
  std::vector<float> samples;
  std::function<void(Canvas &)> paint;
};

struct Context::Panel {
  std::string id, title;
  float x{}, y{}, width{}, height{}, next_y{};
  std::vector<Item> items;
  int row_columns{}, row_index{};
  float row_y{}, row_height{};
};

Context::Context(grassland::graphics::Core *core, grassland::graphics::Window *window, const std::string &font_file)
    : core_(core),
      window_(window) {
  if (!core || !window || font_file.empty())
    throw std::invalid_argument("GUI requires core, window and font");
  draw::CreateCore(core, &painter_);
  painter_->SetFontTypeFile(font_file);
  core_->CreateShader(GetShaderCode("shaders/blur.slang"), "Blur", "cs_6_0", &blur_shader_);
  core_->CreateShader(GetShaderCode("shaders/composite.slang"), "Composite", "cs_6_0", &composite_shader_);
  core_->CreateComputeProgram(blur_shader_.get(), &blur_program_);
  blur_program_->AddResourceBinding(grassland::graphics::RESOURCE_TYPE_IMAGE, 1);
  blur_program_->AddResourceBinding(grassland::graphics::RESOURCE_TYPE_WRITABLE_IMAGE, 1);
  blur_program_->AddResourceBinding(grassland::graphics::RESOURCE_TYPE_UNIFORM_BUFFER, 1);
  blur_program_->Finalize();
  core_->CreateComputeProgram(composite_shader_.get(), &composite_program_);
  composite_program_->AddResourceBinding(grassland::graphics::RESOURCE_TYPE_IMAGE, 1);
  composite_program_->AddResourceBinding(grassland::graphics::RESOURCE_TYPE_IMAGE, 1);
  composite_program_->AddResourceBinding(grassland::graphics::RESOURCE_TYPE_WRITABLE_IMAGE, 1);
  composite_program_->AddResourceBinding(grassland::graphics::RESOURCE_TYPE_UNIFORM_BUFFER, 1);
  composite_program_->Finalize();
  core_->CreateBuffer(sizeof(BlurSettings), grassland::graphics::BUFFER_TYPE_DYNAMIC, &blur_settings_horizontal_);
  core_->CreateBuffer(sizeof(BlurSettings), grassland::graphics::BUFFER_TYPE_DYNAMIC, &blur_settings_vertical_);
  core_->CreateBuffer(sizeof(GlassSettings), grassland::graphics::BUFFER_TYPE_DYNAMIC, &glass_settings_);
}

Context::~Context() = default;

void Context::SetPointerInput(float x, float y, bool down) {
  pointer_override_ = glm::vec3{x, y, down ? 1.0f : 0.0f};
}

void Context::ClearPointerInput() {
  pointer_override_.reset();
}

void Context::SetLogicalSize(int width, int height) {
  logical_override_ = {width, height};
}

void Context::BeginFrame() {
  panels_.clear();
  current_ = nullptr;
  const auto cursor = window_->GetCursorPosition();
  cursor_x_ = pointer_override_ ? pointer_override_->x : static_cast<float>(cursor.x);
  cursor_y_ = pointer_override_ ? pointer_override_->y : static_cast<float>(cursor.y);
  const bool down =
      pointer_override_ ? pointer_override_->z > 0.5f : window_->IsMouseButtonDown(GLFW_MOUSE_BUTTON_LEFT);
  pointer_pressed_ = down && !pointer_down_;
  pointer_released_ = !down && pointer_down_;
  pointer_down_ = down;
  wants_pointer_ = false;
}

bool Context::Hit(float x, float y, float width, float height) const {
  return cursor_x_ >= x && cursor_x_ < x + width && cursor_y_ >= y && cursor_y_ < y + height;
}

bool Context::Activate(const std::string &id, float x, float y, float width, float height) {
  const bool hovered = Hit(x, y, width, height);
  wants_pointer_ |= hovered || (active_id_ == id && pointer_down_);
  if (pointer_pressed_ && hovered)
    active_id_ = id;
  const bool clicked = pointer_released_ && active_id_ == id && hovered;
  if (pointer_released_ && active_id_ == id)
    active_id_.clear();
  return clicked;
}

void Context::BeginPanel(const std::string &id, float x, float y, float width, const std::string &title) {
  if (current_)
    throw std::logic_error("nested GUI panels are not supported");
  panels_.push_back({id, title, x, y, width, 0.0f, y + (title.empty() ? 18.0f : 55.0f), {}});
  current_ = &panels_.back();
}

void Context::EndPanel() {
  if (!current_)
    throw std::logic_error("EndPanel without BeginPanel");
  if (current_->row_columns)
    EndRow();
  current_->height = current_->next_y - current_->y + 14.0f;
  wants_pointer_ |= Hit(current_->x, current_->y, current_->width, current_->height);
  current_ = nullptr;
}

void Context::BeginRow(int columns) {
  if (!current_ || current_->row_columns || columns < 1)
    throw std::logic_error("invalid GUI row");
  current_->row_columns = columns;
  current_->row_index = 0;
  current_->row_y = current_->next_y;
  current_->row_height = 0;
}

void Context::EndRow() {
  if (!current_ || !current_->row_columns)
    throw std::logic_error("EndRow without BeginRow");
  current_->next_y = current_->row_y + current_->row_height + 7.0f;
  current_->row_columns = 0;
}

Context::Frame Context::NextFrame(float height) {
  if (!current_)
    throw std::logic_error("GUI item outside panel");
  if (!current_->row_columns) {
    const float y = current_->next_y;
    current_->next_y += height + 7.0f;
    return {current_->x + 18, y, current_->width - 36};
  }
  if (current_->row_index >= current_->row_columns)
    throw std::logic_error("too many items in GUI row");
  const float width = (current_->width - 36 - (current_->row_columns - 1) * 8.0f) / current_->row_columns;
  const float x = current_->x + 18 + current_->row_index++ * (width + 8.0f);
  current_->row_height = std::max(current_->row_height, height);
  return {x, current_->row_y, width};
}

float Context::NextY(float height) {
  return NextFrame(height).y;
}

void Context::Text(const std::string &text) {
  const auto frame = NextFrame(21);
  current_->items.push_back({Item::Kind::kText, text, frame.x, frame.y, frame.width, 21});
}

void Context::Heading(const std::string &text) {
  const auto frame = NextFrame(29);
  current_->items.push_back({Item::Kind::kHeading, text, frame.x, frame.y, frame.width, 29});
}

void Context::Spacer(float height) {
  NextY(height);
}

void Context::Separator() {
  const auto frame = NextFrame(3);
  current_->items.push_back({Item::Kind::kSeparator, {}, frame.x, frame.y, frame.width, 3});
}

bool Context::Button(const std::string &label) {
  const auto frame = NextFrame(36);
  const float x = frame.x, y = frame.y, w = frame.width;
  const std::string id = current_->id + "/button/" + label;
  const bool clicked = Activate(id, x, y, w, 36);
  current_->items.push_back({Item::Kind::kButton, label, x, y, w, 36, 0, false, Hit(x, y, w, 36), active_id_ == id});
  return clicked;
}

bool Context::Checkbox(const std::string &label, bool *value) {
  const auto frame = NextFrame(31);
  const float x = frame.x, y = frame.y, w = frame.width;
  const bool changed = Activate(current_->id + "/check/" + label, x, y, w, 31);
  if (changed)
    *value = !*value;
  current_->items.push_back({Item::Kind::kCheckbox, label, x, y, w, 31, 0, *value, Hit(x, y, w, 31)});
  return changed;
}

bool Context::Slider(const std::string &label, float *value, float min, float max) {
  const auto frame = NextFrame(51);
  const float x = frame.x, y = frame.y, w = frame.width;
  const std::string id = current_->id + "/slider/" + label;
  Activate(id, x, y, w, 51);
  const float before = *value;
  if (pointer_down_ && active_id_ == id && max > min)
    *value = std::clamp(min + (cursor_x_ - x) / w * (max - min), min, max);
  const float fraction = max > min ? std::clamp((*value - min) / (max - min), 0.0f, 1.0f) : 0.0f;
  current_->items.push_back({Item::Kind::kSlider, ValueLabel(label, *value, max - min), x, y, w, 51, fraction, false,
                             Hit(x, y, w, 51), active_id_ == id});
  return before != *value;
}

bool Context::Slider(const std::string &label, int *value, int min, int max) {
  float temporary = static_cast<float>(*value);
  Slider(label, &temporary, static_cast<float>(min), static_cast<float>(max));
  const int rounded = std::clamp(static_cast<int>(std::round(temporary)), min, max);
  const bool changed = *value != rounded;
  *value = rounded;
  current_->items.back().label = label + "  " + std::to_string(*value);
  return changed;
}

bool Context::Choice(const std::string &label, int *index, const std::vector<std::string> &items) {
  if (items.empty())
    return false;
  const auto frame = NextFrame(37);
  const float x = frame.x, y = frame.y, w = frame.width;
  const bool clicked = Activate(current_->id + "/choice/" + label, x, y, w, 37);
  if (clicked)
    *index = (*index + 1) % static_cast<int>(items.size());
  const int safe = std::clamp(*index, 0, static_cast<int>(items.size()) - 1);
  current_->items.push_back(
      {Item::Kind::kChoice, label + "   " + items[safe] + "  >", x, y, w, 37, 0, false, Hit(x, y, w, 37)});
  return clicked;
}

void Context::Plot(const std::string &label, const std::vector<float> &values, float min, float max) {
  const auto frame = NextFrame(88);
  Item item{Item::Kind::kPlot, label, frame.x, frame.y, frame.width, 88};
  item.samples.reserve(values.size());
  for (float v : values)
    item.samples.push_back(max > min ? std::clamp((v - min) / (max - min), 0.0f, 1.0f) : 0.0f);
  current_->items.push_back(std::move(item));
}

CanvasResponse Context::Custom(const std::string &id, float height, std::function<void(Canvas &)> paint) {
  const auto frame = NextFrame(height);
  const std::string full_id = current_->id + "/canvas/" + id;
  const bool clicked = Activate(full_id, frame.x, frame.y, frame.width, height);
  const bool hovered = Hit(frame.x, frame.y, frame.width, height);
  Item item{Item::Kind::kCanvas, id, frame.x, frame.y, frame.width, height};
  item.paint = std::move(paint);
  current_->items.push_back(std::move(item));
  return {{cursor_x_ - frame.x, cursor_y_ - frame.y}, hovered, pointer_pressed_ && hovered, clicked};
}

void Context::DrawRect(float x, float y, float width, float height, Color color, float radius) {
  if (width <= 0 || height <= 0)
    return;
  const int pixel_width = std::max(1, int(std::round(width * scale_x_)));
  const int pixel_height = std::max(1, int(std::round(height * scale_y_)));
  const int pixel_radius = std::max(0, int(std::round(radius * std::min(scale_x_, scale_y_))));
  const auto key = std::make_tuple(pixel_width, pixel_height, pixel_radius);
  auto found = rect_models_.find(key);
  if (found == rect_models_.end()) {
    std::unique_ptr<draw::Model> model;
    painter_->CreateModel(&model);
    std::vector<draw::Vertex> vertices;
    std::vector<uint32_t> indices;
    constexpr int segments = 8;
    const float corner_radius = std::min({float(pixel_radius), pixel_width * 0.5f, pixel_height * 0.5f});
    const float rx = corner_radius / pixel_width, ry = corner_radius / pixel_height;
    vertices.push_back({{0.5f, 0.5f}, {0.5f, 0.5f}, {1, 1, 1, 1}});
    for (int corner = 0; corner < 4; ++corner) {
      const float cx = corner == 0 || corner == 3 ? rx : 1.0f - rx;
      const float cy = corner < 2 ? ry : 1.0f - ry;
      const float start = 3.14159265f + corner * 1.570796325f;
      for (int i = 0; i <= segments; ++i) {
        const float angle = start + i * 1.570796325f / segments;
        const glm::vec2 p{cx + std::cos(angle) * rx, cy + std::sin(angle) * ry};
        vertices.push_back({p, p, {1, 1, 1, 1}});
        if (vertices.size() > 2)
          indices.insert(indices.end(),
                         {0, static_cast<uint32_t>(vertices.size() - 2), static_cast<uint32_t>(vertices.size() - 1)});
      }
    }
    indices.insert(indices.end(), {0, static_cast<uint32_t>(vertices.size() - 1), 1});
    model->SetModelData(vertices, indices);
    found = rect_models_.emplace(key, std::move(model)).first;
  }
  glm::mat4 transform = glm::translate(glm::mat4(1), glm::vec3(x * scale_x_, y * scale_y_, 0));
  transform = glm::scale(transform, glm::vec3(width * scale_x_, height * scale_y_, 1));
  painter_->CmdDrawInstance(found->second.get(), draw::PixelCoordToNDC(target_width_, target_height_) * transform,
                            Vec(color));
}

void Context::DrawText(float x, float y, const std::string &text, Color color, unsigned size) {
  painter_->SetFontSize(std::max(1u, static_cast<unsigned>(std::round(size * scale_y_))));
  painter_->CmdDrawText({x * scale_x_, y * scale_y_}, text, Vec(color));
}

void Context::PaintItem(const Item &item) {
  switch (item.kind) {
    case Item::Kind::kText:
      DrawText(item.x, item.y + 16, item.label, theme_.muted, 15);
      break;
    case Item::Kind::kHeading:
      DrawText(item.x, item.y + 21, item.label, theme_.text, 18);
      break;
    case Item::Kind::kSeparator:
      DrawRect(item.x, item.y, item.width, 1, {1, 1, 1, 0.20f});
      break;
    case Item::Kind::kButton:
    case Item::Kind::kChoice:
      DrawRect(item.x, item.y, item.width, item.height,
               item.active    ? Alpha(theme_.accent, 0.9f)
               : item.hovered ? Alpha(theme_.control, 1.0f)
                              : theme_.control);
      DrawText(item.x + 12, item.y + 24, item.label, item.active ? Color{1, 1, 1, 1} : theme_.text, 15);
      break;
    case Item::Kind::kCheckbox:
      DrawRect(item.x, item.y + 3, 24, 24, item.checked ? theme_.accent : theme_.control);
      if (item.checked)
        DrawText(item.x + 5, item.y + 22, "✓", {1, 1, 1, 1}, 16);
      DrawText(item.x + 34, item.y + 22, item.label, theme_.text, 15);
      break;
    case Item::Kind::kSlider:
      DrawText(item.x, item.y + 17, item.label, theme_.text, 14);
      DrawRect(item.x, item.y + 35, item.width, 7, theme_.control, 4);
      DrawRect(item.x, item.y + 35, std::max(7.0f, item.width * item.value), 7, theme_.accent, 4);
      DrawRect(item.x + item.width * item.value - 6, item.y + 30, 15, 16,
               theme_.dark ? Color{0.91f, 0.94f, 1.0f, 1} : Color{1, 1, 1, 1}, 8);
      break;
    case Item::Kind::kPlot:
      DrawText(item.x, item.y + 17, item.label, theme_.text, 14);
      DrawRect(item.x, item.y + 23, item.width, 58, Alpha(theme_.control, 0.45f));
      for (size_t i = 0; i < item.samples.size(); ++i) {
        const float w = item.width / std::max<size_t>(1, item.samples.size());
        DrawRect(item.x + i * w, item.y + 80 - item.samples[i] * 54, std::max(1.0f, w - 1),
                 std::max(2.0f, item.samples[i] * 54), theme_.accent);
      }
      break;
    case Item::Kind::kCanvas: {
      Canvas canvas(this, item.x, item.y, item.width, item.height);
      item.paint(canvas);
      break;
    }
  }
}

void Canvas::Rectangle(float x, float y, float width, float height, Color color, float radius) {
  context_->DrawRect(x_ + x, y_ + y, width, height, color, radius);
}

void Canvas::Circle(float x, float y, float radius, Color color) {
  context_->DrawRect(x_ + x - radius, y_ + y - radius, radius * 2, radius * 2, color, radius);
}

void Canvas::Label(float x, float baseline, const std::string &text, Color color, unsigned size) {
  context_->DrawText(x_ + x, y_ + baseline, text, color, size);
}

void Canvas::ParticleField(int count, float time, Color color) {
  for (int i = 0; i < count; ++i) {
    const float phase = i * 2.399963f;
    const float speed = 0.13f + float(i % 7) * 0.018f;
    const float x = std::fmod((i * 0.618034f + time * speed), 1.0f) * width_;
    const float y = std::fmod((i * 0.414214f + time * speed * 0.7f), 1.0f) * height_;
    const float radius = 1.5f + 1.5f * (0.5f + 0.5f * std::sin(phase + time));
    Circle(x, y, radius, color);
  }
}

grassland::graphics::Image *Context::EndFrame(grassland::graphics::CommandContext *commands,
                                              grassland::graphics::Image *scene) {
  if (current_)
    throw std::logic_error("GUI panel not closed");
  const auto logical = logical_override_.x > 0 && logical_override_.y > 0 ? logical_override_ : window_->GetSize();
  const auto physical = scene->Extent();
  if (logical.x <= 0 || logical.y <= 0 || physical.width == 0 || physical.height == 0)
    return scene;
  target_width_ = static_cast<int>(physical.width);
  target_height_ = static_cast<int>(physical.height);
  scale_x_ = static_cast<float>(physical.width) / logical.x;
  scale_y_ = static_cast<float>(physical.height) / logical.y;
  auto *target = scene;
  if (theme_.backdrop_blur && !panels_.empty()) {
    if (!output_ || output_->Extent().width != physical.width || output_->Extent().height != physical.height ||
        output_->Format() != scene->Format()) {
      core_->WaitGPU();
      horizontal_.reset();
      vertical_.reset();
      output_.reset();
      core_->CreateImage(physical.width, physical.height, scene->Format(), &horizontal_);
      core_->CreateImage(physical.width, physical.height, scene->Format(), &vertical_);
      core_->CreateImage(physical.width, physical.height, scene->Format(), &output_);
    }
    const BlurSettings horizontal{{2, 0}, {target_width_, target_height_}};
    const BlurSettings vertical{{0, 2}, {target_width_, target_height_}};
    blur_settings_horizontal_->UploadData(&horizontal, sizeof(horizontal));
    blur_settings_vertical_->UploadData(&vertical, sizeof(vertical));
    commands->CmdBindComputeProgram(blur_program_.get());
    commands->CmdBindResources(0, {scene}, grassland::graphics::BIND_POINT_COMPUTE);
    commands->CmdBindResources(1, {horizontal_.get()}, grassland::graphics::BIND_POINT_COMPUTE);
    commands->CmdBindResources(2, {blur_settings_horizontal_.get()}, grassland::graphics::BIND_POINT_COMPUTE);
    commands->CmdDispatch((physical.width + 7) / 8, (physical.height + 7) / 8, 1);
    commands->CmdBindResources(0, {horizontal_.get()}, grassland::graphics::BIND_POINT_COMPUTE);
    commands->CmdBindResources(1, {vertical_.get()}, grassland::graphics::BIND_POINT_COMPUTE);
    commands->CmdBindResources(2, {blur_settings_vertical_.get()}, grassland::graphics::BIND_POINT_COMPUTE);
    commands->CmdDispatch((physical.width + 7) / 8, (physical.height + 7) / 8, 1);
    GlassSettings glass{};
    glass.extent = {target_width_, target_height_};
    glass.count = std::min<int>(panels_.size(), 16);
    for (int i = 0; i < glass.count; ++i) {
      const auto &panel = panels_[i];
      glass.rects[i] = {panel.x * scale_x_, panel.y * scale_y_, (panel.x + panel.width) * scale_x_,
                        (panel.y + panel.height) * scale_y_};
    }
    glass_settings_->UploadData(&glass, sizeof(glass));
    commands->CmdBindComputeProgram(composite_program_.get());
    commands->CmdBindResources(0, {scene}, grassland::graphics::BIND_POINT_COMPUTE);
    commands->CmdBindResources(1, {vertical_.get()}, grassland::graphics::BIND_POINT_COMPUTE);
    commands->CmdBindResources(2, {output_.get()}, grassland::graphics::BIND_POINT_COMPUTE);
    commands->CmdBindResources(3, {glass_settings_.get()}, grassland::graphics::BIND_POINT_COMPUTE);
    commands->CmdDispatch((physical.width + 7) / 8, (physical.height + 7) / 8, 1);
    target = output_.get();
  }
  painter_->CmdSetDrawRegion(0, 0, target_width_, target_height_);
  for (const auto &panel : panels_) {
    DrawRect(panel.x, panel.y + 5, panel.width, panel.height, {0, 0, 0, theme_.dark ? 0.32f : 0.16f}, 19);
    DrawRect(panel.x, panel.y, panel.width, panel.height, {1, 1, 1, theme_.dark ? 0.18f : 0.38f}, 19);
    DrawRect(panel.x + 1, panel.y + 1, panel.width - 2, panel.height - 2, theme_.panel, 18);
    if (!panel.title.empty())
      DrawText(panel.x + 18, panel.y + 35, panel.title, theme_.text, 21);
    for (const auto &item : panel.items)
      PaintItem(item);
  }
  painter_->Render(commands, target);
  return target;
}
}  // namespace snowberg::gui
