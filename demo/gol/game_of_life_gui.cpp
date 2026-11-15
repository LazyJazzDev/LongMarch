#include "game_of_life_gui.h"

#include <limits>

#ifndef LONGMARCH_HOSTED_GAMES
#include <tinyfiledialogs.h>
#endif

#include <random>
#include <stdexcept>
#include <utility>

#include "application/listener.h"
#include "application/model.h"
#include "cells_pattern.h"
#include "game_of_life_lib.h"
#include "glm/gtc/matrix_transform.hpp"
#include "gpu/tiled_life_storage.h"

namespace life_demo {

namespace {
// Playground background, also the color seen through the gaps between cells.
constexpr float kPlaygroundBrightness = 0.08f;
}  // namespace

GameOfLife::GameOfLife(const char *title,
                       int width,
                       int height,
                       int cell_grid_width,
                       int cell_grid_height,
                       rhi::BackendAPI api,
                       bool hosted)
    : Application(title, width, height, api, hosted),
      cell_grid_width_(cell_grid_width),
      cell_grid_height_(cell_grid_height) {
}

GameOfLife::~GameOfLife() = default;

void GameOfLife::SetInitialCells(std::vector<uint8_t> cells) {
  if (cells.size() != static_cast<size_t>(cell_grid_width_ * cell_grid_height_))
    throw std::invalid_argument("Initial cells must match the grid size");
  initial_cells_ = std::move(cells);
}

void GameOfLife::OnFramebufferResize() {
  OnWindowSize();
}

void GameOfLife::CustomOnInit() {
  {
    std::vector<glm::vec2> positions = {
        {0.0, 0.0},
    };
    std::vector<uint32_t> indices = {};
    const auto num_petal = 120;
    const auto norm_level = 8.0f;

    for (int i = 0; i < num_petal; i++) {
      float theta = float(i) * float(glm::pi<float>() * 2) / float(num_petal);
      auto sin_t = std::sin(theta);
      auto cos_t = std::cos(theta);
      auto norm =
          std::pow(std::pow(std::abs(sin_t), norm_level) + std::pow(std::abs(cos_t), norm_level), 1.0f / norm_level);
      positions.emplace_back(sin_t / norm, cos_t / norm);
      indices.push_back(0);
      indices.push_back(i + 1);
      indices.push_back((i + 1) % num_petal + 1);
    }

    std::vector<Vertex> vertices = ComposeVertices(positions, glm::vec4{1.0f});

    white_icon_model = std::make_optional<DeviceModel>(this, Model(vertices, indices));
  }

  white_rect_model = std::make_optional<DeviceModel>(
      this, Model(ComposeVertices({{-1.0f, -1.0f}, {-1.0f, 1.0f}, {1.0f, -1.0f}, {1.0f, 1.0f}}, glm::vec4{1.0f}),
                  {0, 1, 2, 2, 1, 3}));

  InitCells(cell_grid_width_, cell_grid_height_);
  boundary_button_ = std::make_unique<BoundaryToggleButton>(this, &white_icon_model.value());
  pause_play_button_ = std::make_unique<PausePlayButton>(this, 10.0f, 10.0f, 110.0f, 110.0f, &white_icon_model.value());
  pause_play_button_->SetPlaying(initial_playing_);
  speed_toggle_button_ =
      std::make_unique<SpeedToggleButton>(this, 10.0f, 120.0f, 110.0f, 220.0f, &white_icon_model.value());
  refresh_button_ =
      std::make_unique<RefreshButton>(this, 10.0f, 230.0f, 110.0f, 330.0f, &cell_grid_, &white_icon_model.value());

  randomize_button_ = std::make_unique<RandomizeButton>(this, &cell_grid_, &white_icon_model.value());
  if (tiled_enabled_) {
    refresh_button_->SetResetHandler([this] {
      pending_fill_ = 0;
      pending_pattern_.clear();
      pending_toggles_.clear();
    });
    randomize_button_->SetRandomizeHandler([this] {
      pending_fill_ = 1;
      pending_pattern_.clear();
      pending_toggles_.clear();
    });
  }
  open_button_ = std::make_unique<FileButton>(this, &white_icon_model.value(), FileButton::Kind::kOpen,
                                              [this] { RequestFileAction(FileAction::kOpen); });
  save_button_ = std::make_unique<FileButton>(this, &white_icon_model.value(), FileButton::Kind::kSave,
                                              [this] { RequestFileAction(FileAction::kSave); });

  requested_width_ = cell_grid_width_;
  requested_height_ = cell_grid_height_;
  if (!tiled_enabled_) {
    width_slider_ = std::make_unique<SizeSlider>(this, 'W', cell_grid_width_, &white_rect_model.value(),
                                                 [this](int value) { requested_width_ = value; });
    height_slider_ = std::make_unique<SizeSlider>(this, 'H', cell_grid_height_, &white_rect_model.value(),
                                                  [this](int value) { requested_height_ = value; });
  }
  OnWindowSize();
  magnify_callback_ = GetWindow()->MagnifyEvent().RegisterCallback([this](const rhi::MagnifyGesture &gesture) {
    if (gesture.phase == rhi::MagnifyPhase::kCancel)
      return;
    const glm::vec2 point = FramePosition({gesture.x, gesture.y});
    if (point.x < playground_left_ || point.x >= playground_right_ || point.y < playground_top_ ||
        point.y >= playground_bottom_)
      return;
    const glm::vec2 center{(playground_left_ + playground_right_) * 0.5f,
                           (playground_top_ + playground_bottom_) * 0.5f};
    ruler_timing_.Zoom(grassland::GetTimeSeconds());
    grid_view_.Zoom(float(gesture.scale), point - center, tiled_enabled_ ? 64.0f / TiledLife::kWorldSize : 1.0f);
    LayoutCells();
  });
  scroll_callback_ = GetWindow()->ScrollEvent().RegisterCallback([this](double x, double y) { ScrollGrid(x, y); });
  pan_button_callback_ =
      GetWindow()->MouseButtonEvent().RegisterCallback([this](int button, int action, int, double x, double y) {
        const auto cursor = FramePosition({x, y});
        const bool in_grid = cursor.x >= playground_left_ && cursor.x < playground_right_ &&
                             cursor.y >= playground_top_ && cursor.y < playground_bottom_;
        if (button == GLFW_MOUSE_BUTTON_RIGHT) {
          pan_cursor_ = cursor;
          panning_ = action == GLFW_PRESS && in_grid;
        } else if (button == GLFW_MOUSE_BUTTON_LEFT) {
          // A left drag (one finger on touch screens) pans an enlarged grid; a
          // tap still toggles the cell. The drag becomes a pan once it moves.
          drag_start_ = cursor;
          drag_pending_ = action == GLFW_PRESS && in_grid && (tiled_enabled_ || grid_view_.zoom > 1.0f);
          if (action != GLFW_PRESS)
            EndDragPan();
        }
      });
  pan_move_callback_ = GetWindow()->MouseMoveEvent().RegisterCallback([this](double x, double y) {
    const auto cursor = FramePosition({x, y});
    const glm::vec2 scale = glm::vec2(FramebufferSize()) / glm::vec2(glm::max(GetWindowSize(), glm::ivec2{1}));
    if (drag_pending_ && glm::length(cursor - drag_start_) > 8.0f * scale.x) {
      drag_pending_ = false;
      drag_panning_ = panning_ = true;
      pan_cursor_ = drag_start_;
      cell_input_->SetSuspended(true);
    }
    if (!panning_)
      return;
    grid_view_.pan += cursor - pan_cursor_;
    pan_cursor_ = cursor;
    LayoutCells();
  });
  pan_focus_callback_ = GetWindow()->FocusEvent().RegisterCallback([this](bool focused) {
    if (!focused) {
      panning_ = false;
      EndDragPan();
    }
  });
  key_callback_ = GetWindow()->KeyEvent().RegisterCallback([this](int key, int, int action, int mods) {
    if (action != GLFW_PRESS && action != GLFW_REPEAT)
      return;
    if (!(mods & (GLFW_MOD_CONTROL | GLFW_MOD_SUPER)))
      return;
    if (action == GLFW_PRESS && (key == GLFW_KEY_O || key == GLFW_KEY_S)) {
      RequestFileAction(key == GLFW_KEY_O ? FileAction::kOpen : FileAction::kSave);
      return;
    }
    if (key == GLFW_KEY_0 || key == GLFW_KEY_KP_0) {
      grid_view_ = {};
      LayoutCells();
    } else if (key == GLFW_KEY_EQUAL || key == GLFW_KEY_KP_ADD) {
      ZoomGrid(1.2f);
    } else if (key == GLFW_KEY_MINUS || key == GLFW_KEY_KP_SUBTRACT) {
      ZoomGrid(1.0f / 1.2f);
    }
  });
  last_frame_time_ = last_simulation_time_ = grassland::GetTimeSeconds();
}

void GameOfLife::CustomOnUpdate() {
  // Apply at most one resize per frame, outside listener event dispatch.
  const bool dragging = width_slider_ && (width_slider_->IsDragging() || height_slider_->IsDragging());
  if (requested_width_ != cell_grid_width_ || requested_height_ != cell_grid_height_)
    ResizeGrid(requested_width_, requested_height_);
  else if (sliders_were_dragging_ && !dragging)
    OnWindowSize();
  sliders_were_dragging_ = dragging;

  auto current_frame_time = grassland::GetTimeSeconds();
  auto delta_time = static_cast<float>(current_frame_time - last_frame_time_);
  const double simulation_elapsed = current_frame_time - last_simulation_time_;
  last_frame_time_ = last_simulation_time_ = current_frame_time;

  if (file_action_ != FileAction::kNone) {
    file_action_delay_ -= delta_time;
    if (file_action_delay_ <= 0.0f) {
      ProcessFileAction();
      // Modal dialogs must not accumulate simulation time or skip UI feedback.
      last_frame_time_ = last_simulation_time_ = grassland::GetTimeSeconds();
      delta_time = 0.0f;
    }
  }

  icon_rotation_.Update(delta_time / 0.25f);
  SetIconRotation(float(icon_rotation_));
  time_total += delta_time;

  // Update pause_play_button_
  pause_play_button_->Update(delta_time);
  // Update speed_toggle_button_
  speed_toggle_button_->Update(delta_time);
  boundary_button_->Update(delta_time);
  // Update refresh_button_
  refresh_button_->Update(delta_time);
  randomize_button_->Update(delta_time);
  open_button_->Update(delta_time);
  save_button_->Update(delta_time);

  simulation_clock_.Advance(simulation_elapsed, pause_play_button_->IsPlaying() && file_action_ == FileAction::kNone,
                            speed_toggle_button_->SpeedLevel(), [this] {
                              if (tiled_enabled_)
                                ++pending_steps_;
                              else
                                update_step(cell_grid_width_, cell_grid_height_, cell_grid_.data(),
                                            boundary_button_->Mode());
                            });

  // Synchronize after stepping so the new generation is visible in this frame.
  // One pass over the cells: animate, collect the shader data and remember
  // whether another frame is needed, instead of revisiting 65536 objects.
  cells_animating_ = cell_input_->Update(delta_time, !pause_play_button_->IsPlaying());

  // Draw pause_play_button_
  pause_play_button_->Draw();
  // Draw speed_toggle_button_
  speed_toggle_button_->Draw();
  boundary_button_->Draw();
  // Draw refresh_button_
  refresh_button_->Draw();
  randomize_button_->Draw();
  open_button_->Draw();
  save_button_->Draw();
  if (!tiled_enabled_) {
    width_slider_->Draw();
    height_slider_->Draw();
  }

  DrawCellGrid();
  DrawGridRuler();

  DrawModel(&white_rect_model.value(),
            {GetModelMatrix(glm::vec2{panel_left_, panel_top_},
                            glm::vec2{panel_right_ - panel_left_, panel_bottom_ - panel_top_}, 0.8f),
             glm::vec4{0.1, 0.1, 0.1, 1.0}, glm::uvec4{0}});
  // Match the opposite action rail to the main toolbar background.
  const auto framebuffer = glm::vec2(FramebufferSize());
  const glm::vec2 action_position = sidebar_ ? glm::vec2{playground_right_, 0} : glm::vec2{0, 0};
  const glm::vec2 action_size = sidebar_ ? glm::vec2{framebuffer.x - playground_right_, framebuffer.y}
                                         : glm::vec2{framebuffer.x, playground_top_};
  DrawModel(&white_rect_model.value(),
            {GetModelMatrix(action_position, action_size, 0.8f), glm::vec4{0.1, 0.1, 0.1, 1.0}, glm::uvec4{0}});
  DrawModel(
      &white_rect_model.value(),
      {GetModelMatrix(glm::vec2{playground_left_, playground_top_},
                      glm::vec2{playground_right_ - playground_left_, playground_bottom_ - playground_top_}, 0.8f),
       {kPlaygroundBrightness, kPlaygroundBrightness, kPlaygroundBrightness, 1.0},
       glm::uvec4{0}});
}

void GameOfLife::CustomOnClose() {
  GetWindow()->FocusEvent().UnregisterCallback(pan_focus_callback_);
  GetWindow()->MouseButtonEvent().UnregisterCallback(pan_button_callback_);
  GetWindow()->MouseMoveEvent().UnregisterCallback(pan_move_callback_);
  GetWindow()->MagnifyEvent().UnregisterCallback(magnify_callback_);
  GetWindow()->ScrollEvent().UnregisterCallback(scroll_callback_);
  GetWindow()->KeyEvent().UnregisterCallback(key_callback_);
  // Release all resources
  width_slider_.reset();
  height_slider_.reset();
  cell_input_.reset();
  pause_play_button_.reset();
  speed_toggle_button_.reset();
  boundary_button_.reset();
  refresh_button_.reset();
  randomize_button_.reset();
  open_button_.reset();
  save_button_.reset();
  white_rect_model.reset();
  white_icon_model.reset();
}

void GameOfLife::EndDragPan() {
  drag_pending_ = false;
  if (!drag_panning_)
    return;
  drag_panning_ = panning_ = false;
  cell_input_->SetSuspended(false);
}

void GameOfLife::OnWindowSize() {
  panning_ = false;
  EndDragPan();
  auto window_width = float(FramebufferSize().x);
  auto window_height = float(FramebufferSize().y);
  auto extent = std::min(window_width, window_height);
  if (control_extent_limit_ > 0.0f)
    extent = std::min(extent, window_height * control_extent_limit_);
  auto ui_unit = extent * 0.01f * ui_scale_;
  if (control_pixel_size_ > 0.0f)
    ui_unit = std::min(control_pixel_size_ * 0.1f, std::min(window_width, window_height) / 90.0f);
  const float margin = ui_unit * 5.0f;
  // Native hosts reserve the home-gesture strip without shrinking the canvas.
  const float bottom_margin = std::max(margin, window_height * bottom_control_inset_);
  const float bottom_shift = bottom_margin - margin;
  const float icon_size = ui_unit * 10.0f;
  const float step = ui_unit * 13.0f;
  const float slider_gap = ui_unit * 1.5f;
  const float slider_thickness = (icon_size - slider_gap) * 0.5f;
  const float panel_size = ui_unit * 20.0f;
  // Controls keep at least a margin from a host's safe-area boundary (a screen
  // cutout); without host safe areas they use the fallback gap.
  const bool cutouts = cutout_insets_.y >= 0.0f;
  auto edge_offset = [&](float inset, float fallback) { return cutouts ? std::max(margin, inset) : fallback; };

  float playground_left = 0.0f;
  float playground_right = window_width;
  float playground_top = 0.0f;
  float playground_bottom = window_height;
  panel_left_ = 0;
  panel_right_ = window_width;
  panel_top_ = 0;
  panel_bottom_ = window_height;

  // Sidebars follow the window's short edges, independent of the grid aspect.
  sidebar_ = window_width >= window_height;
  auto place = [icon_size](Button *button, float x, float y) { button->Resize(x, y, x + icon_size, y + icon_size); };
  if (sidebar_) {
    const float left_x = edge_offset(window_width * cutout_insets_.x, margin);
    const float right_x = window_width - edge_offset(window_width * cutout_insets_.z, margin) - icon_size;
    playground_left = left_x + icon_size + margin;
    playground_right = right_x - margin;
    panel_right_ = playground_left;
    place(open_button_.get(), left_x, margin);
    place(save_button_.get(), left_x, margin + step);
    place(boundary_button_.get(), left_x, window_height - bottom_margin - icon_size - step * 2.0f);
    place(speed_toggle_button_.get(), left_x, window_height - bottom_margin - icon_size - step);
    place(pause_play_button_.get(), left_x, window_height - bottom_margin - icon_size);
    place(refresh_button_.get(), right_x, margin);
    place(randomize_button_.get(), right_x, window_height - bottom_margin - icon_size);
    const float top = margin + step * 2.0f;
    const float bottom = window_height - top - step - bottom_shift;
    if (width_slider_) {
      width_slider_->Resize({left_x, top, left_x + slider_thickness, bottom}, true);
      height_slider_->Resize({left_x + slider_thickness + slider_gap, top, left_x + icon_size, bottom}, true);
    }
  } else {
    // The sparse top actions sit in the corners beside a top cutout, so they
    // center on its safe-area boundary. Without host safe areas, give them
    // more breathing room against rounded corners.
    const float action_top_margin = edge_offset(window_height * cutout_insets_.y - icon_size * 0.5f, margin * 2.0f);
    playground_top = panel_size + action_top_margin - margin;
    playground_bottom = window_height - panel_size - bottom_shift;
    panel_top_ = playground_bottom;
    const float top = window_height - bottom_margin - icon_size;
    place(open_button_.get(), margin, top);
    place(save_button_.get(), margin + step, top);
    place(boundary_button_.get(), window_width - margin - icon_size - step * 2.0f, top);
    place(speed_toggle_button_.get(), window_width - margin - icon_size - step, top);
    place(pause_play_button_.get(), window_width - margin - icon_size, top);
    // Keep the sparse action rail at the top, leaving its center clear for
    // phone camera cutouts. The dense controls occupy the opposite short edge.
    place(refresh_button_.get(), margin, action_top_margin);
    place(randomize_button_.get(), window_width - margin - icon_size, action_top_margin);
    const float left = margin + step * 2.0f;
    const float right = window_width - left - step;
    if (width_slider_) {
      width_slider_->Resize({left, top, right, top + slider_thickness}, false);
      height_slider_->Resize({left, top + slider_thickness + slider_gap, right, top + icon_size}, false);
    }
  }

  if (tiled_enabled_) {
    if (!sidebar_) {
      const float top = window_height - bottom_margin - icon_size;
      const float spacing = (window_width - 2 * margin - icon_size) / 4;
      Button *buttons[] = {open_button_.get(), save_button_.get(), boundary_button_.get(), speed_toggle_button_.get(),
                           pause_play_button_.get()};
      for (int i = 0; i < 5; ++i)
        place(buttons[i], margin + i * spacing, top);
    }
  }
  playground_left_ = playground_left;
  playground_right_ = playground_right;
  playground_top_ = playground_top;
  playground_bottom_ = playground_bottom;

  LayoutCells();
}

void GameOfLife::LayoutCells() {
  const glm::vec2 viewport{playground_right_ - playground_left_, playground_bottom_ - playground_top_};
  const float fitted_unit = tiled_enabled_
                                ? std::min(viewport.x, viewport.y) * (0.95f / 64.0f)
                                : std::min(viewport.y / cell_grid_height_, viewport.x / cell_grid_width_) * 0.95f;
  if (tiled_enabled_) {
    const auto limit = glm::max(
        (glm::vec2{TiledLife::kWorldSize} * fitted_unit * grid_view_.zoom - viewport * 0.9f) * 0.5f, glm::vec2{0});
    grid_view_.pan = glm::clamp(grid_view_.pan, -limit, limit);
  } else
    grid_view_.Clamp(viewport, glm::vec2{cell_grid_width_, cell_grid_height_} * fitted_unit);
  const float cell_unit = fitted_unit * grid_view_.zoom;
  const glm::vec2 origin = glm::vec2{float(playground_left_ + playground_right_) * 0.5f,
                                     float(playground_bottom_ + playground_top_) * 0.5f} -
                           (tiled_enabled_ ? glm::vec2{TiledLife::kWorldSize}
                                           : glm::vec2{float(cell_grid_width_), float(cell_grid_height_)}) *
                               cell_unit * 0.5f +
                           grid_view_.pan;
  cell_input_->Layout(origin, cell_unit, {playground_left_, playground_top_, playground_right_, playground_bottom_});
}

void GameOfLife::DrawCellGrid() {
  // One quad covers the visible part of the grid; the pixel shader finds the
  // cell under each fragment.
  const glm::vec2 grid_origin = cell_input_->Origin();
  const glm::vec2 grid_end =
      grid_origin + (tiled_enabled_ ? glm::vec2{TiledLife::kWorldSize}
                                    : glm::vec2{float(cell_grid_width_), float(cell_grid_height_)}) *
                        cell_input_->Unit();
  const glm::vec2 low = glm::max(grid_origin, glm::vec2{playground_left_, playground_top_});
  const glm::vec2 high = glm::min(grid_end, glm::vec2{playground_right_, playground_bottom_});
  if (high.x <= low.x || high.y <= low.y)
    return;
  // color carries the grid origin, pitch and gap brightness (the playground
  // background below); extra.zw carries its dimensions.
  if (tiled_enabled_) {
    DrawModel(
        &white_rect_model.value(),
        {GetModelMatrix(low, high - low, 0.4f), glm::vec4{grid_origin, cell_input_->Unit(), kPlaygroundBrightness},
         glm::uvec4{
             6u,
             GridDetail::At(cell_input_->Unit(), float(std::min(FramebufferSize().x, FramebufferSize().y))).Packed(),
             TiledLife::kWorldSize, TiledLife::kWorldSize},
         glm::vec4{playground_left_, playground_top_, playground_right_, playground_bottom_}});
    return;
  }
  DrawModelWithData(
      &white_rect_model.value(),
      {GetModelMatrix(low, high - low, 0.4f), glm::vec4{grid_origin, cell_input_->Unit(), kPlaygroundBrightness},
       glm::uvec4{2u, 0u, uint32_t(cell_grid_width_), uint32_t(cell_grid_height_)}},
      cell_input_->Appearance());
}

void GameOfLife::DrawGridRuler() {
  if (!tiled_enabled_)
    return;
  const float alpha = ruler_timing_.Opacity(grassland::GetTimeSeconds());
  if (alpha <= 0)
    return;
  const glm::vec2 low{playground_left_, playground_top_};
  const glm::vec2 size{playground_right_ - playground_left_, playground_bottom_ - playground_top_};
  const float font =
      std::max(12.0f, control_pixel_size_ > 0 ? control_pixel_size_ * 0.25f : std::min(size.x, size.y) * 0.022f);
  DrawModel(
      &white_rect_model.value(),
      {GetModelMatrix(low, size, 0.2f), glm::vec4{cell_input_->Origin(), cell_input_->Unit(), alpha},
       glm::uvec4{
           7u, GridDetail::At(cell_input_->Unit(), float(std::min(FramebufferSize().x, FramebufferSize().y))).Packed(),
           uint32_t(font * 64), 0u},
       glm::vec4{low, low + size}});
}

glm::vec2 GameOfLife::CursorPosition() const {
  return FramePosition(GetCursorPosition());
}

glm::vec2 GameOfLife::FramePosition(glm::dvec2 position) const {
  return glm::vec2(position) * glm::vec2(FramebufferSize()) / glm::vec2(glm::max(GetWindowSize(), glm::ivec2{1}));
}

bool GameOfLife::CursorInGrid() const {
  auto p = CursorPosition();
  return p.x >= playground_left_ && p.x < playground_right_ && p.y >= playground_top_ && p.y < playground_bottom_;
}

void GameOfLife::ZoomGrid(float factor) {
  ruler_timing_.Zoom(grassland::GetTimeSeconds());
  const glm::vec2 center{(playground_left_ + playground_right_) * 0.5f, (playground_top_ + playground_bottom_) * 0.5f};
  grid_view_.Zoom(factor, CursorInGrid() ? CursorPosition() - center : glm::vec2{0.0f},
                  tiled_enabled_ ? 64.0f / TiledLife::kWorldSize : 1.0f);
  LayoutCells();
}

void GameOfLife::ScrollGrid(double x, double y) {
  if (!CursorInGrid())
    return;
  auto down = [this](int key) { return GetWindow()->IsKeyDown(key); };
  if (down(GLFW_KEY_LEFT_CONTROL) || down(GLFW_KEY_RIGHT_CONTROL)) {
    ZoomGrid(std::exp(float(y) * 0.08f));
    return;
  }
  if ((down(GLFW_KEY_LEFT_SHIFT) || down(GLFW_KEY_RIGHT_SHIFT)) && x == 0.0)
    std::swap(x, y);
  const glm::vec2 scale = glm::vec2(FramebufferSize()) / glm::vec2(glm::max(GetWindowSize(), glm::ivec2{1}));
  grid_view_.pan += glm::vec2{float(x), float(y)} * scale * 10.0f;
  LayoutCells();
}

void GameOfLife::ResizeGrid(int width, int height) {
  width = std::clamp(width, grid_size::kMin, grid_size::kMax);
  height = std::clamp(height, grid_size::kMin, grid_size::kMax);
  auto resized = grid_size::Resize(cell_grid_, cell_grid_width_, cell_grid_height_, width, height);
  cell_grid_ = std::move(resized);
  cell_grid_width_ = width;
  cell_grid_height_ = height;
  // Rebind after vector reallocation.
  if (!tiled_enabled_)
    cell_input_->Reset(cell_grid_.data(), cell_grid_width_, cell_grid_height_);
  grid_view_ = {};
  OnWindowSize();
}

void GameOfLife::RequestFileAction(FileAction action) {
  if (file_action_ != FileAction::kNone)
    return;
  file_action_ = action;
  (action == FileAction::kSave ? save_button_.get() : open_button_.get())->BeginAction();
  // Let the button start moving before opening a blocking native dialog.
  file_action_delay_ = 0.12f;
}

void GameOfLife::ProcessFileAction() {
#ifdef LONGMARCH_HOSTED_GAMES
  hosted_file_action_ = file_action_ == FileAction::kSave ? 2 : 1;
  file_action_ = FileAction::kNone;
  hosted_was_playing_ = pause_play_button_->IsPlaying();
  pause_play_button_->SetPlaying(false);
#else
  const auto action = std::exchange(file_action_, FileAction::kNone);
  auto *button = action == FileAction::kSave ? save_button_.get() : open_button_.get();
  const bool was_playing = pause_play_button_->IsPlaying();
  pause_play_button_->SetPlaying(false);
  bool loaded = false;
#ifdef _WIN32
  tinyfd_winUtf8 = 1;
#endif
  const char *filters[] = {"*.cells"};
  try {
    const char *selection =
        action == FileAction::kSave
            ? tinyfd_saveFileDialog("Save current grid", file_path_.c_str(), 1, filters, "Life pattern (*.cells)")
            : tinyfd_openFileDialog("Open a Life grid", file_path_.c_str(), 1, filters, "Life pattern (*.cells)", 0);
    if (selection) {
      // Dialog results share an internal buffer; own the path before another call.
      std::string path(selection);
      bool accepted = true;
      if (action == FileAction::kSave) {
        if (std::filesystem::u8path(path).extension().empty()) {
          path += ".cells";
          if (std::filesystem::exists(std::filesystem::u8path(path)))
            accepted = tinyfd_messageBox("Replace saved grid?", "The .cells file already exists. Replace it?", "yesno",
                                         "question", 0) == 1;
        }
        if (accepted)
          SaveCellsPattern(path, {cell_grid_width_, cell_grid_height_, cell_grid_});
      } else {
        auto pattern = FitCellsPattern(LoadCellsPattern(path), cell_grid_width_, cell_grid_height_);
        const int width = pattern.width;
        const int height = pattern.height;
        const auto &cells = pattern.cells;
        ResizeGrid(width, height);
        // Copy into the bound storage; cell buttons retain pointers to these cells.
        std::copy(cells.begin(), cells.end(), cell_grid_.begin());
        requested_width_ = width;
        requested_height_ = height;
        if (width_slider_)
          width_slider_->SetValue(width);
        if (height_slider_)
          height_slider_->SetValue(height);
        if (tiled_enabled_)
          QueueCenteredPattern();
        loaded = true;
      }
      if (accepted) {
        file_path_ = std::move(path);
        button->Feedback(true);
      }
    }
  } catch (const std::exception &error) {
    LogError("Grid file operation failed: {}", error.what());
    // tinyfiledialogs may use shell-backed dialogs; keep their message text literal.
    const char *message =
        action == FileAction::kSave
            ? "Could not save the grid. Check that the folder exists, is writable, and has free disk space."
            : "Could not read the grid. Choose a readable .cells file under 1 MiB, with equal-length rows of O and . and at most 256 rows and columns.";
    tinyfd_messageBox(action == FileAction::kSave ? "Could not save grid" : "Could not open grid", message, "ok",
                      "error", 1);
    button->Feedback(false);
  }
  pause_play_button_->SetPlaying(loaded ? false : was_playing);
  simulation_clock_ = {};
  open_button_->OnCursorEnter(0);
  save_button_->OnCursorEnter(0);
  FocusWindow();
#endif
}

std::string GameOfLife::CompleteHostedFile(const std::string &path) {
  if (!hosted_file_action_)
    return "No pending file operation";
  auto *button = hosted_file_action_ == 2 ? save_button_.get() : open_button_.get();
  bool loaded = false;
  std::string error;
  try {
    if (!path.empty()) {
      int tiled_load = 0;
      if (tiled_enabled_ && hosted_file_action_ == 1)
        tiled_load = LoadTiledState(path, error);
      if (hosted_file_action_ == 2) {
        if (tiled_enabled_)
          error = SaveTiledState(path);
        else
          SaveCellsPattern(path, {cell_grid_width_, cell_grid_height_, cell_grid_});
      } else if (tiled_load == 0) {
        auto pattern = FitCellsPattern(LoadCellsPattern(path), cell_grid_width_, cell_grid_height_);
        ResizeGrid(pattern.width, pattern.height);
        std::copy(pattern.cells.begin(), pattern.cells.end(), cell_grid_.begin());
        requested_width_ = pattern.width;
        requested_height_ = pattern.height;
        if (width_slider_)
          width_slider_->SetValue(pattern.width);
        if (height_slider_)
          height_slider_->SetValue(pattern.height);
        if (tiled_enabled_)
          QueueCenteredPattern();
        loaded = true;
      } else
        loaded = tiled_load == 1;
      button->Feedback(error.empty());
    }
  } catch (const std::exception &e) {
    error = e.what();
    button->Feedback(false);
  }
  hosted_file_action_ = 0;
  ResetFrameClock();
  pause_play_button_->SetPlaying(loaded ? false : hosted_was_playing_);
  simulation_clock_ = {};
  open_button_->OnCursorEnter(0);
  save_button_->OnCursorEnter(0);
  return error;
}

void GameOfLife::RandomizeCells(float density, uint32_t seed) {
  std::mt19937 random_engine(seed);
  std::bernoulli_distribution alive(density);
  for (auto &cell : cell_grid_) {
    cell = alive(random_engine) ? 1 : 0;
  }
}

void GameOfLife::InitCells(int width, int height) {
  cell_grid_width_ = width;
  cell_grid_height_ = height;
  cell_grid_.resize(cell_grid_width_ * cell_grid_height_);
  if (!initial_cells_.empty()) {
    cell_grid_ = std::move(initial_cells_);
  } else if (random_density_ > 0.0f) {
    RandomizeCells(random_density_, random_seed_);
  }
  cell_input_ = std::make_unique<CellGrid>(this);
  if (tiled_enabled_) {
    tiled_ready_ = tiled_life_.Initialize(Device(), ShaderSource("shaders/tiled_life.slang"));
    if (tiled_ready_)
      SetCellStorage(tiled_life_.Cells(), tiled_life_.Mipmaps());
    cell_input_->ResetGPU(TiledLife::kWorldSize, TiledLife::kWorldSize,
                          [this](int x, int y) { pending_toggles_.push_back({uint32_t(x), uint32_t(y)}); });
    QueueCenteredPattern();
  } else
    cell_input_->Reset(cell_grid_.data(), cell_grid_width_, cell_grid_height_);
}

// The experimental full-world format is identified by a versioned magic header.
// Ordinary text .cells patterns remain supported for importing small designs.
std::string GameOfLife::SaveTiledState(const std::string &path) {
  rhi::ObjPtr<rhi::ICommandContext> commands = Queue()->CreateCommandContext();
  if (!commands || !PrepareScene(commands.Get()))
    return "Could not prepare the grid for saving";
  std::vector<uint32_t> packed(tiled_life_.WordCount());
  rhi::ObjPtr<rhi::IDownloadToken> download = commands->CmdDownloadBuffer(tiled_life_.Cells(), 0, packed.size() * 4);
  rhi::ObjPtr<rhi::ISubmissionToken> submitted = commands->Submit();
  if (!download || !submitted || !submitted->WaitHost() || !download->CopyTo(packed.data(), packed.size() * 4))
    return "Could not read the grid for saving";
  std::ofstream output(std::filesystem::u8path(path), std::ios::binary | std::ios::trunc);
  std::string error;
  if (!WriteTiledState(output, packed, error))
    return error;
  output.close();
  return output ? std::string{} : "Could not save the grid";
}

int GameOfLife::LoadTiledState(const std::string &path, std::string &error) {
  std::ifstream input(std::filesystem::u8path(path), std::ios::binary);
  std::array<char, 8> magic{};
  input.read(magic.data(), magic.size());
  if (std::string(magic.data(), magic.size()) != kTiledStateMagic)
    return 0;
  input.clear();
  input.seekg(0);
  std::vector<uint32_t> packed;
  if (!ReadTiledState(input, packed, error))
    return -1;
  pending_pattern_ = std::move(packed);
  pending_fill_ = -1;
  pending_toggles_.clear();
  pending_steps_ = 0;
  grid_view_ = {};
  LayoutCells();
  return 1;
}

void GameOfLife::QueueCenteredPattern() {
  pending_fill_ = -1;
  pending_toggles_.clear();
  pending_pattern_.assign(TiledLife::kWorldSize * TiledLife::kWorldSize / 32, 0);
  const uint32_t ox = (TiledLife::kWorldSize - cell_grid_width_) / 2;
  const uint32_t oy = (TiledLife::kWorldSize - cell_grid_height_) / 2;
  for (uint32_t y = 0; y < uint32_t(cell_grid_height_); ++y) {
    for (uint32_t x = 0; x < uint32_t(cell_grid_width_); ++x) {
      const uint32_t wx = x + ox, wy = y + oy;
      const uint32_t word = ((wy / 256) * 64 + wx / 256) * 2048 + (wy % 256) * 8 + (wx % 256) / 32;
      pending_pattern_[word] |= uint32_t(cell_grid_[y * cell_grid_width_ + x] != 0) << (wx % 32);
    }
  }
}

bool GameOfLife::PrepareScene(rhi::ICommandContext *commands) {
  if (!tiled_enabled_)
    return true;
  if (!tiled_ready_)
    return false;
  if (!pending_pattern_.empty()) {
    if (!tiled_life_.Upload(commands, pending_pattern_))
      return false;
    pending_pattern_.clear();
  }
  if (pending_fill_ >= 0) {
    if (!tiled_life_.Fill(commands, pending_fill_ != 0, uint32_t(std::random_device{}())))
      return false;
    pending_fill_ = -1;
  }
  for (auto cell : pending_toggles_)
    if (!tiled_life_.Toggle(commands, cell.x, cell.y))
      return false;
  pending_toggles_.clear();
  for (uint32_t i = 0; i < pending_steps_; ++i)
    if (!tiled_life_.Step(commands, boundary_button_->Mode() == BoundaryMode::kPeriodic))
      return false;
  pending_steps_ = 0;
  return cell_input_->Unit() >= 1.0f || tiled_life_.UpdateMipmaps(commands, GridDetail::Heat(cell_input_->Unit()));
}

void GameOfLife::EnableNativeSizeControls() {
  if (tiled_enabled_)
    return;
  width_slider_->SetActivationHandler([this] { hosted_size_axis_ = 1; });
  height_slider_->SetActivationHandler([this] { hosted_size_axis_ = 2; });
}

glm::ivec2 GameOfLife::TakeSizeControlRequest() {
  if (tiled_enabled_)
    return {0, 0};
  const int axis = std::exchange(hosted_size_axis_, 0);
  return {axis, axis == 1 ? width_slider_->Value() : height_slider_->Value()};
}

glm::vec4 GameOfLife::SizeControlBounds(int axis) const {
  if (tiled_enabled_)
    return glm::vec4{0};
  const glm::vec2 size = glm::max(glm::vec2(FramebufferSize()), glm::vec2{1.0f});
  return (axis == 1 ? width_slider_ : height_slider_)->Bounds() / glm::vec4{size, size};
}

void GameOfLife::SetGridDimension(int axis, int value) {
  if (tiled_enabled_)
    return;
  if (axis == 1)
    width_slider_->SetValue(value);
  else if (axis == 2)
    height_slider_->SetValue(value);
}

void GameOfLife::SetBottomControlInset(float height_fraction) {
  const float inset = std::clamp(height_fraction, 0.0f, 0.25f);
  if (bottom_control_inset_ != inset) {
    bottom_control_inset_ = inset;
    OnWindowSize();
  }
}

void GameOfLife::SetCutoutInsets(float left, float top, float right) {
  const glm::vec3 insets = glm::clamp(glm::vec3{left, top, right}, 0.0f, 0.25f);
  if (cutout_insets_ != insets) {
    cutout_insets_ = insets;
    OnWindowSize();
  }
}

void GameOfLife::SetControlExtentLimit(float height_fraction) {
  const float limit = std::max(height_fraction, 0.0f);
  if (control_extent_limit_ != limit) {
    control_extent_limit_ = limit;
    OnWindowSize();
  }
}

void GameOfLife::SetControlPixelSize(float pixels) {
  const float size = std::isfinite(pixels) ? std::max(pixels, 0.0f) : 0.0f;
  if (control_pixel_size_ != size) {
    control_pixel_size_ = size;
    OnWindowSize();
  }
}

void GameOfLife::SetIconOrientation(float radians) {
  ResetAnimationClock();
  const float current = float(icon_rotation_);
  icon_rotation_.TryUpdateTarget(current + std::remainder(radians - current, glm::two_pi<float>()));
}

double GameOfLife::NextFrameDelay() const {
  if (!icon_rotation_.IsFinished() || file_action_ != FileAction::kNone || pause_play_button_->IsAnimating() ||
      speed_toggle_button_->IsAnimating() || boundary_button_->IsAnimating() || refresh_button_->IsAnimating() ||
      randomize_button_->IsAnimating() || open_button_->IsAnimating() || save_button_->IsAnimating())
    return 0;
  if (cells_animating_)
    return 0;
  const double normal = pause_play_button_->IsPlaying()
                            ? simulation_clock_.NextStepDelay(speed_toggle_button_->SpeedLevel())
                            : std::numeric_limits<double>::infinity();
  return tiled_enabled_ ? ruler_timing_.NextDelay(grassland::GetTimeSeconds(), normal) : normal;
}
}  // namespace life_demo
