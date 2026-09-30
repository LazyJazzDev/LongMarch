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

namespace {
// Playground background, also the color seen through the gaps between cells.
constexpr float kPlaygroundBrightness = 0.08f;
}  // namespace

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
  white_rect_model = std::make_optional<DeviceModel>(
      this, Model(ComposeVertices({{-1.0f, -1.0f}, {-1.0f, 1.0f}, {1.0f, -1.0f}, {1.0f, 1.0f}}, glm::vec4{1.0f}),
                  {0, 1, 2, 2, 1, 3}));

  InitCells(cell_grid_width_, cell_grid_height_);
  requested_width_ = cell_grid_width_;
  requested_height_ = cell_grid_height_;
  playing_ = initial_playing_;
  ui_ = std::make_unique<snowberg::gui::Context>(Core(), GetWindow(), snowberg::gui::DefaultFont());
  auto &style = ui_->Style();
  style.dark = true;
  style.panel = {0.09f, 0.12f, 0.16f, 0.74f};
  style.control = {0.22f, 0.27f, 0.34f, 0.82f};
  style.text = {0.95f, 0.97f, 1.0f, 1.0f};
  style.muted = {0.68f, 0.75f, 0.84f, 1.0f};
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
  if (requested_width_ != cell_grid_width_ || requested_height_ != cell_grid_height_)
    ResizeGrid(requested_width_, requested_height_);

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

  simulation_clock_.Advance(delta_time, playing_ && file_action_ == FileAction::kNone, speed_level_, [this] {
    update_step(cell_grid_width_, cell_grid_height_, cell_grid_.data(), boundary_mode_);
  });

  // Synchronize after stepping so the new generation is visible in this frame.
  // One pass over the cells animates them and collects the shader data.
  cell_input_->Update(delta_time, !playing_);

  DrawCellGrid();

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
  ui_.reset();
  cell_input_.reset();
  white_rect_model.reset();
}

graphics::Image *GameOfLife::ComposeUI(graphics::CommandContext *commands, graphics::Image *scene) {
  ui_->BeginFrame();
  const float width = std::min(288.0f, std::max(220.0f, float(GetWindow()->GetSize().x) - 32.0f));
  ui_->BeginPanel("life", 16.0f, 16.0f, width, "Game of Life");
  ui_->Text(playing_ ? "Simulation running" : "Simulation paused");
  if (ui_->Button(playing_ ? "Pause" : "Play"))
    playing_ = !playing_;
  ui_->BeginRow(2);
  if (ui_->Button("Clear"))
    std::fill(cell_grid_.begin(), cell_grid_.end(), 0);
  if (ui_->Button("Random"))
    RandomizeCells(0.5f, std::random_device{}());
  ui_->EndRow();
  ui_->Choice("Speed", &speed_level_, {"Normal", "2×", "5×", "Frame"});
  bool wrap = boundary_mode_ == BoundaryMode::kPeriodic;
  if (ui_->Checkbox("Wrap edges", &wrap))
    boundary_mode_ = wrap ? BoundaryMode::kPeriodic : BoundaryMode::kFixed;
  ui_->BeginRow(2);
  if (ui_->Button("Open"))
    RequestFileAction(FileAction::kOpen);
  if (ui_->Button("Save"))
    RequestFileAction(FileAction::kSave);
  ui_->EndRow();
  ui_->EndPanel();

  ui_->BeginPanel("grid", 16.0f, 344.0f, width, "Grid size");
  ui_->Slider("Width", &requested_width_, grid_size::kMin, grid_size::kMax);
  ui_->Slider("Height", &requested_height_, grid_size::kMin, grid_size::kMax);
  ui_->EndPanel();
  return ui_->EndFrame(commands, scene);
}

void GameOfLife::OnWindowSize() {
  panning_ = false;
  const auto framebuffer = glm::vec2(FramebufferSize());
  const auto logical = glm::vec2(glm::max(GetWindow()->GetSize(), glm::ivec2{1}));
  const auto scale = framebuffer / logical;
  const bool side_panel = logical.x >= 700.0f;
  playground_left_ = side_panel ? std::min(framebuffer.x * 0.45f, 324.0f * scale.x) : 0.0f;
  playground_right_ = framebuffer.x;
  playground_top_ = side_panel ? 0.0f : std::min(framebuffer.y * 0.7f, 510.0f * scale.y);
  playground_bottom_ = framebuffer.y;
  LayoutCells();
}

void GameOfLife::LayoutCells() {
  const glm::vec2 viewport{playground_right_ - playground_left_, playground_bottom_ - playground_top_};
  const float fitted_unit = std::min(viewport.y / cell_grid_height_, viewport.x / cell_grid_width_) * 0.95f;
  grid_view_.Clamp(viewport, glm::vec2{cell_grid_width_, cell_grid_height_} * fitted_unit);
  const float cell_unit = fitted_unit * grid_view_.zoom;
  const glm::vec2 origin = glm::vec2{float(playground_left_ + playground_right_) * 0.5f,
                                     float(playground_bottom_ + playground_top_) * 0.5f} -
                           glm::vec2{float(cell_grid_width_), float(cell_grid_height_)} * cell_unit * 0.5f +
                           grid_view_.pan;
  cell_input_->Layout(origin, cell_unit, {playground_left_, playground_top_, playground_right_, playground_bottom_});
}

void GameOfLife::DrawCellGrid() {
  // One quad covers the visible part of the grid; the pixel shader finds the
  // cell under each fragment.
  const glm::vec2 grid_origin = cell_input_->Origin();
  const glm::vec2 grid_end =
      grid_origin + glm::vec2{float(cell_grid_width_), float(cell_grid_height_)} * cell_input_->Unit();
  const glm::vec2 low = glm::max(grid_origin, glm::vec2{playground_left_, playground_top_});
  const glm::vec2 high = glm::min(grid_end, glm::vec2{playground_right_, playground_bottom_});
  if (high.x <= low.x || high.y <= low.y)
    return;
  // color carries the grid origin, pitch and gap brightness (the playground
  // background below); extra.zw carries its dimensions.
  DrawModelWithData(
      &white_rect_model.value(),
      {GetModelMatrix(low, high - low, 0.4f), glm::vec4{grid_origin, cell_input_->Unit(), kPlaygroundBrightness},
       glm::uvec4{2u, 0u, uint32_t(cell_grid_width_), uint32_t(cell_grid_height_)}},
      cell_input_->Appearance());
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
  // Rebind after vector reallocation.
  cell_input_->Reset(cell_grid_.data(), cell_grid_width_, cell_grid_height_);
  grid_view_ = {};
  OnWindowSize();
}

void GameOfLife::RequestFileAction(FileAction action) {
  if (file_action_ != FileAction::kNone)
    return;
  file_action_ = action;
  // Allow one frame of UI feedback before opening a blocking native dialog.
  file_action_delay_ = 0.12f;
}

void GameOfLife::ProcessFileAction() {
  const auto action = std::exchange(file_action_, FileAction::kNone);
  const bool was_playing = playing_;
  playing_ = false;
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
        // Copy into the storage bound to the cell renderer.
        std::copy(cells.begin(), cells.end(), cell_grid_.begin());
        requested_width_ = width;
        requested_height_ = height;
        loaded = true;
      }
      if (accepted) {
        file_path_ = std::move(path);
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
  }
  playing_ = loaded ? false : was_playing;
  simulation_clock_ = {};
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
  cell_input_ = std::make_unique<CellGrid>(this);
  cell_input_->Reset(cell_grid_.data(), cell_grid_width_, cell_grid_height_);
}
