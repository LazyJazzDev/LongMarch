#include "game_of_life_gui.h"

#include <tinyfiledialogs.h>

#include <random>
#include <stdexcept>
#include <utility>

#include "application/listener.h"
#include "application/model.h"
#include "cells_pattern.h"
#include "game_of_life_lib.h"
#include "glm/gtc/matrix_transform.hpp"

GameOfLife::GameOfLife(const char *title,
                       int width,
                       int height,
                       int cell_grid_width,
                       int cell_grid_height,
                       graphics::BackendAPI api)
    : Application(title, width, height, api),
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
  pause_play_button_ = std::make_unique<PausePlayButton>(this, 10.0f, 10.0f, 110.0f, 110.0f, &white_icon_model.value());
  pause_play_button_->SetPlaying(initial_playing_);
  speed_toggle_button_ =
      std::make_unique<SpeedToggleButton>(this, 10.0f, 120.0f, 110.0f, 220.0f, &white_icon_model.value());
  refresh_button_ =
      std::make_unique<RefreshButton>(this, 10.0f, 230.0f, 110.0f, 330.0f, &cell_grid_, &white_icon_model.value());

  randomize_button_ = std::make_unique<RandomizeButton>(this, &cell_grid_, &white_icon_model.value());
  open_button_ = std::make_unique<FileButton>(this, &white_icon_model.value(), FileButton::Kind::kOpen,
                                              [this] { RequestFileAction(FileAction::kOpen); });
  save_button_ = std::make_unique<FileButton>(this, &white_icon_model.value(), FileButton::Kind::kSave,
                                              [this] { RequestFileAction(FileAction::kSave); });

  requested_width_ = cell_grid_width_;
  requested_height_ = cell_grid_height_;
  width_slider_ = std::make_unique<SizeSlider>(this, 'W', cell_grid_width_, &white_rect_model.value(),
                                               [this](int value) { requested_width_ = value; });
  height_slider_ = std::make_unique<SizeSlider>(this, 'H', cell_grid_height_, &white_rect_model.value(),
                                                [this](int value) { requested_height_ = value; });
  OnWindowSize();
  magnify_callback_ = GetWindow()->MagnifyEvent().RegisterCallback([this](const graphics::MagnifyGesture &gesture) {
    if (gesture.phase == graphics::MagnifyPhase::kCancel)
      return;
    const glm::vec2 point = FramePosition({gesture.x, gesture.y});
    if (point.x < playground_left_ || point.x >= playground_right_ || point.y < playground_top_ ||
        point.y >= playground_bottom_)
      return;
    const glm::vec2 center{(playground_left_ + playground_right_) * 0.5f,
                           (playground_top_ + playground_bottom_) * 0.5f};
    grid_view_.Zoom(float(gesture.scale), point - center);
    LayoutCells();
  });
  scroll_callback_ = GetWindow()->ScrollEvent().RegisterCallback([this](double x, double y) { ScrollGrid(x, y); });
  pan_button_callback_ =
      GetWindow()->MouseButtonEvent().RegisterCallback([this](int button, int action, int, double x, double y) {
        if (button != GLFW_MOUSE_BUTTON_RIGHT)
          return;
        pan_cursor_ = FramePosition({x, y});
        panning_ = action == GLFW_PRESS && pan_cursor_.x >= playground_left_ && pan_cursor_.x < playground_right_ &&
                   pan_cursor_.y >= playground_top_ && pan_cursor_.y < playground_bottom_;
      });
  pan_move_callback_ = GetWindow()->MouseMoveEvent().RegisterCallback([this](double x, double y) {
    if (!panning_)
      return;
    const auto cursor = FramePosition({x, y});
    grid_view_.pan += cursor - pan_cursor_;
    pan_cursor_ = cursor;
    LayoutCells();
  });
  pan_focus_callback_ = GetWindow()->FocusEvent().RegisterCallback([this](bool focused) {
    if (!focused)
      panning_ = false;
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
  last_frame_time_ = grassland::GetTimeSeconds();
}

void GameOfLife::CustomOnUpdate() {
  // Apply at most one resize per frame, outside listener event dispatch.
  const bool dragging = width_slider_->IsDragging() || height_slider_->IsDragging();
  if (requested_width_ != cell_grid_width_ || requested_height_ != cell_grid_height_)
    ResizeGrid(requested_width_, requested_height_);
  else if (sliders_were_dragging_ && !dragging)
    OnWindowSize();
  sliders_were_dragging_ = dragging;

  auto current_frame_time = grassland::GetTimeSeconds();
  auto delta_time = static_cast<float>(current_frame_time - last_frame_time_);
  last_frame_time_ = current_frame_time;

  if (file_action_ != FileAction::kNone) {
    file_action_delay_ -= delta_time;
    if (file_action_delay_ <= 0.0f) {
      ProcessFileAction();
      // Modal dialogs must not accumulate simulation time or skip UI feedback.
      last_frame_time_ = grassland::GetTimeSeconds();
      delta_time = 0.0f;
    }
  }

  time_total += delta_time;

  // Update pause_play_button_
  pause_play_button_->Update(delta_time);
  // Update speed_toggle_button_
  speed_toggle_button_->Update(delta_time);
  // Update refresh_button_
  refresh_button_->Update(delta_time);
  randomize_button_->Update(delta_time);
  open_button_->Update(delta_time);
  save_button_->Update(delta_time);

  simulation_clock_.Advance(delta_time, pause_play_button_->IsPlaying() && file_action_ == FileAction::kNone,
                            speed_toggle_button_->SpeedLevel(),
                            [this] { update_step(cell_grid_width_, cell_grid_height_, cell_grid_.data()); });

  // Synchronize after stepping so the new generation is visible in this frame.
  for (auto &cell : cell_button_grid_)
    cell->Update(delta_time, !pause_play_button_->IsPlaying());

  // Draw pause_play_button_
  pause_play_button_->Draw();
  // Draw speed_toggle_button_
  speed_toggle_button_->Draw();
  // Draw refresh_button_
  refresh_button_->Draw();
  randomize_button_->Draw();
  open_button_->Draw();
  save_button_->Draw();
  width_slider_->Draw();
  height_slider_->Draw();

  for (int x = 0; x < cell_grid_width_; x++) {
    for (int y = 0; y < cell_grid_height_; y++) {
      int index = y * cell_grid_width_ + x;
      cell_button_grid_[index]->Draw();
    }
  }

  DrawModel(&white_rect_model.value(),
            {GetModelMatrix(glm::vec2{panel_left_, panel_top_},
                            glm::vec2{panel_right_ - panel_left_, panel_bottom_ - panel_top_}, 0.8f),
             glm::vec4{0.1, 0.1, 0.1, 1.0}, glm::uvec4{0}});
  // Match the opposite action rail to the main toolbar background.
  const auto framebuffer = glm::vec2(FramebufferSize());
  const glm::vec2 action_position = sidebar_ ? glm::vec2{playground_right_, 0} : glm::vec2{0, playground_bottom_};
  const glm::vec2 action_size = sidebar_ ? glm::vec2{framebuffer.x - playground_right_, framebuffer.y}
                                         : glm::vec2{framebuffer.x, framebuffer.y - playground_bottom_};
  DrawModel(&white_rect_model.value(),
            {GetModelMatrix(action_position, action_size, 0.8f), glm::vec4{0.1, 0.1, 0.1, 1.0}, glm::uvec4{0}});
  DrawModel(
      &white_rect_model.value(),
      {GetModelMatrix(glm::vec2{playground_left_, playground_top_},
                      glm::vec2{playground_right_ - playground_left_, playground_bottom_ - playground_top_}, 0.8f),
       {0.08, 0.08, 0.08, 1.0},
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
  cell_button_grid_.clear();
  pause_play_button_.reset();
  speed_toggle_button_.reset();
  refresh_button_.reset();
  randomize_button_.reset();
  open_button_.reset();
  save_button_.reset();
  white_rect_model.reset();
  white_icon_model.reset();
}

void GameOfLife::OnWindowSize() {
  panning_ = false;
  auto window_width = float(FramebufferSize().x);
  auto window_height = float(FramebufferSize().y);
  auto ui_unit = std::min(window_width, window_height) * 0.01f * ui_scale_;
  const float margin = ui_unit * 5.0f;
  const float icon_size = ui_unit * 10.0f;
  const float step = ui_unit * 13.0f;
  const float slider_gap = ui_unit * 1.5f;
  const float slider_thickness = (icon_size - slider_gap) * 0.5f;
  const float panel_size = ui_unit * 20.0f;

  float playground_left = 0.0f;
  float playground_right = window_width;
  float playground_top = 0.0f;
  float playground_bottom = window_height;
  panel_left_ = 0;
  panel_right_ = window_width;
  panel_top_ = 0;
  panel_bottom_ = window_height;

  const bool prefer_sidebar =
      std::min(window_height / cell_grid_height_, (window_width - panel_size * 2.0f) / cell_grid_width_) >
      std::min((window_height - panel_size * 2.0f) / cell_grid_height_, window_width / cell_grid_width_);
  if (!width_slider_->IsDragging() && !height_slider_->IsDragging())
    sidebar_ = prefer_sidebar;
  auto place = [icon_size](Button *button, float x, float y) { button->Resize(x, y, x + icon_size, y + icon_size); };
  if (sidebar_) {
    playground_left = panel_size;
    playground_right = window_width - panel_size;
    panel_right_ = panel_size;
    place(open_button_.get(), margin, margin);
    place(save_button_.get(), margin, margin + step);
    place(speed_toggle_button_.get(), margin, window_height - margin - icon_size - step);
    place(pause_play_button_.get(), margin, window_height - margin - icon_size);
    place(refresh_button_.get(), window_width - margin - icon_size, margin);
    place(randomize_button_.get(), window_width - margin - icon_size, window_height - margin - icon_size);
    const float top = margin + step * 2.0f;
    const float bottom = window_height - top;
    width_slider_->Resize({margin, top, margin + slider_thickness, bottom}, true);
    height_slider_->Resize({margin + slider_thickness + slider_gap, top, margin + icon_size, bottom}, true);
  } else {
    playground_top = panel_size;
    playground_bottom = window_height - panel_size;
    panel_bottom_ = playground_top;
    const float top = margin;
    place(open_button_.get(), margin, top);
    place(save_button_.get(), margin + step, top);
    place(speed_toggle_button_.get(), window_width - margin - icon_size - step, top);
    place(pause_play_button_.get(), window_width - margin - icon_size, top);
    place(refresh_button_.get(), margin, window_height - margin - icon_size);
    place(randomize_button_.get(), window_width - margin - icon_size, window_height - margin - icon_size);
    const float left = margin + step * 2.0f;
    const float right = window_width - left;
    width_slider_->Resize({left, top, right, top + slider_thickness}, false);
    height_slider_->Resize({left, top + slider_thickness + slider_gap, right, top + icon_size}, false);
  }

  playground_left_ = playground_left;
  playground_right_ = playground_right;
  playground_top_ = playground_top;
  playground_bottom_ = playground_bottom;

  LayoutCells();
}

void GameOfLife::LayoutCells() {
  const glm::vec2 viewport{playground_right_ - playground_left_, playground_bottom_ - playground_top_};
  const float fitted_unit = std::min(viewport.y / cell_grid_height_, viewport.x / cell_grid_width_) * 0.95f;
  grid_view_.Clamp(viewport, glm::vec2{cell_grid_width_, cell_grid_height_} * fitted_unit);
  const float cell_unit = fitted_unit * grid_view_.zoom;
  float cell_size = cell_unit * 0.8f;
  float cell_gap = (cell_unit - cell_size) * 0.5f;

  for (int x = 0; x < cell_grid_width_; x++) {
    for (int y = 0; y < cell_grid_height_; y++) {
      int index = y * cell_grid_width_ + x;
      float origin_x = float(playground_left_ + playground_right_) * 0.5f - float(cell_grid_width_) * cell_unit * 0.5f +
                       float(x) * cell_unit + grid_view_.pan.x;
      float origin_y = float(playground_bottom_ + playground_top_) * 0.5f -
                       float(cell_grid_height_) * cell_unit * 0.5f + float(y) * cell_unit + grid_view_.pan.y;
      cell_button_grid_[index]->SetClipBounds(
          {playground_left_, playground_top_, playground_right_, playground_bottom_});
      cell_button_grid_[index]->Resize(origin_x + cell_gap, origin_y + cell_gap, origin_x + cell_gap + cell_size,
                                       origin_y + cell_gap + cell_size);
    }
  }
}

glm::vec2 GameOfLife::CursorPosition() const {
  return FramePosition(GetWindow()->GetCursorPosition());
}

glm::vec2 GameOfLife::FramePosition(glm::dvec2 position) const {
  return glm::vec2(position) * glm::vec2(FramebufferSize()) /
         glm::vec2(glm::max(GetWindow()->GetSize(), glm::ivec2{1}));
}

bool GameOfLife::CursorInGrid() const {
  auto p = CursorPosition();
  return p.x >= playground_left_ && p.x < playground_right_ && p.y >= playground_top_ && p.y < playground_bottom_;
}

void GameOfLife::ZoomGrid(float factor) {
  const glm::vec2 center{(playground_left_ + playground_right_) * 0.5f, (playground_top_ + playground_bottom_) * 0.5f};
  grid_view_.Zoom(factor, CursorInGrid() ? CursorPosition() - center : glm::vec2{0.0f});
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
  const glm::vec2 scale = glm::vec2(FramebufferSize()) / glm::vec2(glm::max(GetWindow()->GetSize(), glm::ivec2{1}));
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
  // Reuse existing buttons, then rebind every pointer after vector reallocation.
  if (cell_button_grid_.size() > cell_grid_.size())
    cell_button_grid_.resize(cell_grid_.size());
  for (size_t i = 0; i < cell_grid_.size(); ++i) {
    if (i == cell_button_grid_.size())
      cell_button_grid_.push_back(
          std::make_unique<CellButton>(this, 0, 0, 100, 100, &cell_grid_[i], &white_icon_model.value()));
    cell_button_grid_[i]->Rebind(&cell_grid_[i]);
  }
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
        auto pattern = LoadCellsPattern(path);
        const int width = std::max(pattern.width, grid_size::kMin);
        const int height = std::max(pattern.height, grid_size::kMin);
        auto cells = CenterCellsPattern(pattern, width, height);
        ResizeGrid(width, height);
        // Copy into the bound storage; cell buttons retain pointers to these cells.
        std::copy(cells.begin(), cells.end(), cell_grid_.begin());
        requested_width_ = width;
        requested_height_ = height;
        width_slider_->SetValue(width);
        height_slider_->SetValue(height);
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
            : "Could not read the grid. Choose a readable .cells file under 1 MiB, with equal-length rows of O and . and at most 200 rows and columns.";
    tinyfd_messageBox(action == FileAction::kSave ? "Could not save grid" : "Could not open grid", message, "ok",
                      "error", 1);
    button->Feedback(false);
  }
  pause_play_button_->SetPlaying(loaded ? false : was_playing);
  simulation_clock_ = {};
  open_button_->OnCursorEnter(0);
  save_button_->OnCursorEnter(0);
  GetWindow()->Focus();
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
  for (int y = 0; y < cell_grid_height_; y++) {
    for (int x = 0; x < cell_grid_width_; x++) {
      int index = y * cell_grid_width_ + x;
      cell_button_grid_.push_back(
          std::make_unique<CellButton>(this, 0, 0, 100, 100, &cell_grid_[index], &white_icon_model.value()));
    }
  }
}
