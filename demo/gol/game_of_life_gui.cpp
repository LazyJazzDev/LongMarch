#include "game_of_life_gui.h"

#include <random>

#include "application/listener.h"
#include "application/model.h"
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
  speed_toggle_button_ =
      std::make_unique<SpeedToggleButton>(this, 10.0f, 120.0f, 110.0f, 220.0f, &white_icon_model.value());
  refresh_button_ =
      std::make_unique<RefreshButton>(this, 10.0f, 230.0f, 110.0f, 330.0f, &cell_grid_, &white_icon_model.value());

  randomize_button_ = std::make_unique<RandomizeButton>(this, &cell_grid_, &white_icon_model.value());

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
    const glm::vec2 point = glm::vec2{gesture.x, gesture.y} * glm::vec2(FramebufferSize()) /
                            glm::vec2{std::max(GetWindow()->GetWidth(), 1), std::max(GetWindow()->GetHeight(), 1)};
    if (point.x < playground_left_ || point.x >= playground_right_ || point.y < playground_top_ ||
        point.y >= playground_bottom_)
      return;
    const glm::vec2 center{(playground_left_ + playground_right_) * 0.5f,
                           (playground_top_ + playground_bottom_) * 0.5f};
    grid_view_.Zoom(float(gesture.scale), point - center);
    LayoutCells();
  });
  scroll_callback_ = GetWindow()->ScrollEvent().RegisterCallback([this](double x, double y) { ScrollGrid(x, y); });
  key_callback_ = GetWindow()->KeyEvent().RegisterCallback([this](int key, int, int action, int mods) {
    if (action != GLFW_PRESS && action != GLFW_REPEAT)
      return;
    if (!(mods & GLFW_MOD_CONTROL))
      return;
    if (key == GLFW_KEY_0 || key == GLFW_KEY_KP_0) {
      grid_view_ = {};
      LayoutCells();
    } else if (key == GLFW_KEY_EQUAL || key == GLFW_KEY_KP_ADD) {
      ZoomGrid(1.2f);
    } else if (key == GLFW_KEY_MINUS || key == GLFW_KEY_KP_SUBTRACT) {
      ZoomGrid(1.0f / 1.2f);
    }
  });
}

void GameOfLife::CustomOnUpdate() {
  // Apply at most one resize per frame, outside listener event dispatch.
  const bool dragging = width_slider_->IsDragging() || height_slider_->IsDragging();
  if (requested_width_ != cell_grid_width_ || requested_height_ != cell_grid_height_)
    ResizeGrid(requested_width_, requested_height_);
  else if (sliders_were_dragging_ && !dragging)
    OnWindowSize();
  sliders_were_dragging_ = dragging;

  // Use static timestamp get last frame time (in second, float)
  static auto last_frame_time = glfwGetTime();
  auto current_frame_time = glfwGetTime();
  auto delta_time = static_cast<float>(current_frame_time - last_frame_time);
  last_frame_time = current_frame_time;

  time_total += delta_time;

  // Update pause_play_button_
  pause_play_button_->Update(delta_time);
  // Update speed_toggle_button_
  speed_toggle_button_->Update(delta_time);
  // Update refresh_button_
  refresh_button_->Update(delta_time);
  randomize_button_->Update(delta_time);

  simulation_clock_.Advance(
      delta_time, pause_play_button_->IsPlaying(), speed_toggle_button_->SpeedLevel(),
      [this] { update_step(cell_grid_width_, cell_grid_height_, cell_grid_.data()); }, [] { return glfwGetTime(); });

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
  DrawModel(
      &white_rect_model.value(),
      {GetModelMatrix(glm::vec2{playground_left_, playground_top_},
                      glm::vec2{playground_right_ - playground_left_, playground_bottom_ - playground_top_}, 0.8f),
       {0.08, 0.08, 0.08, 1.0},
       glm::uvec4{0}});
}

void GameOfLife::CustomOnClose() {
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
  white_rect_model.reset();
  white_icon_model.reset();
}

void GameOfLife::OnWindowSize() {
  auto window_width = float(FramebufferSize().x);
  auto window_height = float(FramebufferSize().y);
  auto ui_unit = std::min(window_width, window_height) * 0.01f * ui_scale_;
  float blank_size = ui_unit * 5.0f;
  float icon_size = ui_unit * 10.0f;
  float icon_gap = ui_unit * 3.0f;

  float playground_left = 0.0f;
  float playground_right = window_width;
  float playground_top = 0.0f;
  float playground_bottom = window_height;

  panel_left_ = playground_left;
  panel_right_ = playground_right;
  panel_top_ = playground_top;
  panel_bottom_ = playground_bottom;

  const bool prefer_sidebar =
      std::min((playground_bottom - playground_top) / float(cell_grid_height_),
               (playground_right - playground_left - ui_unit * 20.0f) / float(cell_grid_width_)) >
      std::min((playground_bottom - playground_top - ui_unit * 20.0f) / float(cell_grid_height_),
               (playground_right - playground_left) / float(cell_grid_width_));
  if (!width_slider_->IsDragging() && !height_slider_->IsDragging())
    sidebar_ = prefer_sidebar;
  if (sidebar_) {
    playground_left += ui_unit * 20.0f;
    panel_right_ = playground_left;

    pause_play_button_->Resize(blank_size, window_height - blank_size - icon_size, blank_size + icon_size,
                               window_height - blank_size);
    speed_toggle_button_->Resize(blank_size, window_height - blank_size - icon_size - (icon_size + icon_gap),
                                 blank_size + icon_size, window_height - blank_size - (icon_size + icon_gap));
    refresh_button_->Resize(blank_size, blank_size, blank_size + icon_size, blank_size + icon_size);
    randomize_button_->Resize(blank_size, blank_size + icon_size + icon_gap, blank_size + icon_size,
                              blank_size + icon_size * 2.0f + icon_gap);
  } else {
    playground_bottom -= ui_unit * 20.0f;
    panel_top_ = playground_bottom;

    pause_play_button_->Resize(blank_size, window_height - blank_size - icon_size, blank_size + icon_size,
                               window_height - blank_size);
    speed_toggle_button_->Resize(blank_size + icon_size + icon_gap, window_height - blank_size - icon_size,
                                 blank_size + icon_size * 2.0f + icon_gap, window_height - blank_size);
    refresh_button_->Resize(window_width - blank_size - icon_size, window_height - blank_size - icon_size,
                            window_width - blank_size, window_height - blank_size);
    randomize_button_->Resize(window_width - blank_size - icon_size * 2.0f - icon_gap,
                              window_height - blank_size - icon_size, window_width - blank_size - icon_size - icon_gap,
                              window_height - blank_size);
  }

  const float slider_gap = ui_unit * 1.5f;
  const float thickness = (icon_size - slider_gap) * 0.5f;
  if (sidebar_) {
    const float top = blank_size + icon_size * 2.0f + icon_gap * 2.0f;
    const float bottom = window_height - top;
    width_slider_->Resize({blank_size, top, blank_size + thickness, bottom}, true);
    height_slider_->Resize({blank_size + thickness + slider_gap, top, blank_size + icon_size, bottom}, true);
  } else {
    const float left = blank_size + icon_size * 2.0f + icon_gap * 2.0f;
    const float right = window_width - left;
    const float top = window_height - blank_size - icon_size;
    width_slider_->Resize({left, top, right, top + thickness}, false);
    height_slider_->Resize({left, top + thickness + slider_gap, right, top + icon_size}, false);
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
  double x, y;
  int width, height;
  glfwGetCursorPos(GLFWWindow(), &x, &y);
  glfwGetWindowSize(GLFWWindow(), &width, &height);
  return glm::vec2{float(x), float(y)} * glm::vec2(FramebufferSize()) /
         glm::vec2{std::max(width, 1), std::max(height, 1)};
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
  auto down = [this](int key) { return glfwGetKey(GLFWWindow(), key) == GLFW_PRESS; };
  if (down(GLFW_KEY_LEFT_CONTROL) || down(GLFW_KEY_RIGHT_CONTROL)) {
    ZoomGrid(std::exp(float(y) * 0.08f));
    return;
  }
  if ((down(GLFW_KEY_LEFT_SHIFT) || down(GLFW_KEY_RIGHT_SHIFT)) && x == 0.0)
    std::swap(x, y);
  int width, height;
  glfwGetWindowSize(GLFWWindow(), &width, &height);
  const glm::vec2 scale = glm::vec2(FramebufferSize()) / glm::vec2{std::max(width, 1), std::max(height, 1)};
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
  if (random_density_ > 0.0f) {
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
