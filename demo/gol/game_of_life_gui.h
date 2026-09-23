#pragma once
#include "application/animation_var.h"
#include "application/application.h"
#include "application/model.h"
#include "cell_button.h"
#include "grid_view.h"
#include "pause_play_button.h"
#include "randomize_button.h"
#include "refresh_button.h"
#include "speed_toggle_button.h"

class GameOfLife : public Application {
 public:
  GameOfLife(const char *title,
             int width,
             int height,
             int cell_grid_width,
             int cell_grid_height,
             graphics::BackendAPI api);

  ~GameOfLife() override;

  // Fills the grid with random live cells, each alive with the given probability.
  void RandomizeCells(float density, uint32_t seed);

  // Requests a random initial grid; applied when the cells are created.
  void SetRandomInitialCells(float density, uint32_t seed) {
    random_density_ = density;
    random_seed_ = seed;
  }

 private:
  void CustomOnInit() override;
  void CustomOnUpdate() override;
  void CustomOnClose() override;
  void OnFramebufferResize() override;

  void OnWindowSize();

  void InitCells(int width, int height);
  void LayoutCells();
  glm::vec2 CursorPosition() const;
  bool CursorInGrid() const;
  void ZoomGrid(float factor);
  void ScrollGrid(double x, double y);

  GridView grid_view_;
  uint32_t scroll_callback_{};
  uint32_t key_callback_{};
#ifdef __APPLE__
  void *gesture_monitor_{};
#endif

  std::unique_ptr<PausePlayButton> pause_play_button_;
  std::unique_ptr<SpeedToggleButton> speed_toggle_button_;
  std::unique_ptr<RefreshButton> refresh_button_;
  std::unique_ptr<RandomizeButton> randomize_button_;

  std::optional<DeviceModel> white_icon_model;
  std::optional<DeviceModel> white_rect_model;

  std::vector<uint8_t> cell_grid_;
  std::vector<std::unique_ptr<CellButton>> cell_button_grid_;
  float time_total{0.0};
  float ui_scale_{1.0};
  float random_density_{0.0f};
  uint32_t random_seed_{0};
  int cell_grid_width_{};
  int cell_grid_height_{};

  float panel_left_{};
  float panel_right_{};
  float panel_top_{};
  float panel_bottom_{};

  float playground_left_{};
  float playground_right_{};
  float playground_top_{};
  float playground_bottom_{};
};
