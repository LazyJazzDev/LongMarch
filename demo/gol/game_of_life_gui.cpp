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

  OnWindowSize();
}

void GameOfLife::CustomOnUpdate() {
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

  switch (speed_toggle_button_->SpeedLevel()) {
    case 1:
      delta_time *= 2.0;
      break;
    case 2:
      delta_time *= 5.0;
      break;
    default:
      break;
  }
  for (int x = 0; x < cell_grid_width_; x++) {
    for (int y = 0; y < cell_grid_height_; y++) {
      int index = y * cell_grid_width_ + x;
      cell_button_grid_[index]->Update(delta_time);
    }
  }

  static float accumulate_time = 0.0;
  if (pause_play_button_->IsPlaying()) {
    accumulate_time += delta_time;
    float cost = 0.5;
    while (accumulate_time > cost) {
      update_step(cell_grid_width_, cell_grid_height_, cell_grid_.data());
      accumulate_time -= cost;
    }
  } else {
    accumulate_time = 0.0;
  }

  // Draw pause_play_button_
  pause_play_button_->Draw();
  // Draw speed_toggle_button_
  speed_toggle_button_->Draw();
  // Draw refresh_button_
  refresh_button_->Draw();
  randomize_button_->Draw();

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
  // Release all resources
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

  if (std::min((playground_bottom - playground_top) / float(cell_grid_height_),
               (playground_right - playground_left - ui_unit * 20.0f) / float(cell_grid_width_)) >
      std::min((playground_bottom - playground_top - ui_unit * 20.0f) / float(cell_grid_height_),
               (playground_right - playground_left) / float(cell_grid_width_))) {
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

  playground_left_ = playground_left;
  playground_right_ = playground_right;
  playground_top_ = playground_top;
  playground_bottom_ = playground_bottom;

  float cell_unit = std::min((playground_bottom - playground_top) / float(cell_grid_height_),
                             (playground_right - playground_left) / float(cell_grid_width_)) *
                    0.95;
  float cell_size = cell_unit * 0.8f;
  float cell_gap = (cell_unit - cell_size) * 0.5f;

  for (int x = 0; x < cell_grid_width_; x++) {
    for (int y = 0; y < cell_grid_height_; y++) {
      int index = y * cell_grid_width_ + x;
      float origin_x = float(playground_left + playground_right) * 0.5f - float(cell_grid_width_) * cell_unit * 0.5f +
                       float(x) * cell_unit;
      float origin_y = float(playground_bottom + playground_top) * 0.5f - float(cell_grid_height_) * cell_unit * 0.5f +
                       float(y) * cell_unit;
      cell_button_grid_[index]->Resize(origin_x + cell_gap, origin_y + cell_gap, origin_x + cell_gap + cell_size,
                                       origin_y + cell_gap + cell_size);
    }
  }
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
